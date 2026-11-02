#!/usr/bin/env python3
"""Build/run PTO communication payloads against independent host golden bytes.

This validates tile arithmetic and memory accesses, not a remote transport.
Missing tools, missing/short exports, timeout and simulator errors all fail.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
ARCH_ROOT = CASE_ROOT.parents[2]
NAME = "solution_communication_comm_payload"
CAPACITY = 96
COUNTS = (32, 64, 96)


def input_and_golden():
    # Dyadic FP32 values: both multiply and sum are exactly representable.
    inputs = [[(r + 1) * 8 + ((i * 7) % 31 - 15) / 8 for i in range(CAPACITY)]
              for r in range(3)]
    weights = (0.5, -1.0, 2.0)
    golden = []
    for op in range(4):
        for count in COUNTS:
            for i in range(CAPACITY):
                if i >= count:
                    value = -9999.0
                elif op < 2:
                    value = inputs[op][i]
                elif op == 2:
                    value = sum(row[i] for row in inputs)
                else:
                    value = sum(row[i] * weight for row, weight in zip(inputs, weights))
                golden.append(value)
    return inputs, weights, struct.pack(f"<{len(golden)}f", *golden)


def verify_exports(case_dir, golden):
    expected_status = bytes(4 * 4 * len(COUNTS))
    for name, expected in (("outputs.bin", golden), ("statuses.bin", expected_status)):
        actual = (case_dir / name).read_bytes()
        if actual != expected:
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
            raise ValueError(f"{name}: {len(actual)} bytes, expected {len(expected)}; "
                             f"first differing byte={mismatch}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", required=True, type=Path)
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    case_dir = ARCH_ROOT / "compare" / NAME
    case_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "scope": "single-PE PTO payload; no remote transport"}
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive")
        compiler = args.compiler_dir.resolve() / "clang++"
        model = args.gfrun.resolve()
        if not compiler.is_file() or not model.is_file():
            raise FileNotFoundError("compiler or gfrun is missing")
        report["compiler"] = subprocess.check_output([str(compiler), "--version"], text=True).strip()
        report["gfrun"] = str(model)
        report["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CASE_ROOT, text=True).strip()
        report["worktree_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=CASE_ROOT, text=True).splitlines()
        sources = [*sorted((ARCH_ROOT / "kernels/communication").glob("*.hpp")),
                   CASE_ROOT / "src/comm_payload.cpp", Path(__file__).resolve(),
                   CASE_ROOT / "Makefile", ARCH_ROOT / "test/common/Makefile.common",
                   ARCH_ROOT / "test/common/src/group_worker_runtime.c"]
        report["source_sha256"] = {
            str(path.relative_to(ARCH_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}
        report["gfrun_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        report["compiler_sha256"] = hashlib.sha256(compiler.read_bytes()).hexdigest()
        for name in ("outputs.bin", "statuses.bin"):
            (case_dir / name).unlink(missing_ok=True)
        inputs, weights, golden = input_and_golden()
        flat = [x for row in inputs for x in row]
        (case_dir / "inputs.bin").write_bytes(struct.pack(f"<{len(flat)}f", *flat))
        (case_dir / "weights.bin").write_bytes(struct.pack("<3f", *weights))
        (case_dir / "golden.bin").write_bytes(golden)
        build = ["make", "TESTCASE=comm_payload", "res_check=on",
                 f"COMPILER_DIR={compiler.parent}", "diss"]
        report["build_command"] = build
        with (case_dir / "build.log").open("w") as log:
            subprocess.run(build, cwd=CASE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        elf = ARCH_ROOT / "output/solution/communication/elf" / (NAME + ".elf")
        report["elf_sha256"] = hashlib.sha256(elf.read_bytes()).hexdigest()
        command = [str(model), "-s", "softcore.multiThreadNum=1", "-f", str(elf)]
        report["run_command"] = command
        with (case_dir / "gfrun.log").open("w") as log:
            subprocess.run(command, cwd=CASE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        log_text = (case_dir / "gfrun.log").read_text(errors="replace")
        if not re.search(r"Reach the End of Benchmark[^\n]*R2\s*=\s*0\b", log_text):
            raise ValueError("missing successful model finisher")
        verify_exports(case_dir, golden)
        report.update(status="PASS", cases=12, compared_bytes=len(golden),
                      coverage=["put", "get", "reduce_sum", "weighted_combine", "untouched tails"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    (case_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
