#!/usr/bin/env python3
"""Build and run active microbenchmarks with deterministic classification."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "benchmark"))
from native_bstart import contains
DEFAULT_COMPILER = Path("/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin")
DEFAULT_GFRUN = Path("/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun")


def expected_opcode(case: dict) -> str:
    if case.get("command_mnemonic") == "BSTART.TIMG2COL":
        return "BSTART.TLSU TIMG2COL"
    return canonical_operation(case["operation"])


def canonical_operation(operation: str) -> str:
    return operation.upper().replace("_MX", "MX").replace("_", ".")


DTYPE = {
    "bf16": "BF16", "fp16": "FP16", "fp32": "FP32",
    "i8": "S8", "i16": "S16", "i32": "S32",
    "u8": "U8", "u32": "U32", "s4x2": "S4X2",
}


def canonical_dtype(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    if dtype not in DTYPE:
        raise ValueError("unknown microbenchmark dtype: " + dtype)
    return DTYPE[dtype]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-dir", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gfrun", type=Path, default=DEFAULT_GFRUN)
    parser.add_argument("--category", choices=["all", "cube", "vector", "memory", "scalar", "fixp"], default="all")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--res-check", action="store_true",
                        help="build and run the in-kernel numerical checks")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", type=Path, default=REPO / "output/microbenchmark/report")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build_rc = 0
    if not args.skip_build:
        env = os.environ.copy()
        env["COMPILER_DIR"] = str(args.compiler_dir)
        env["res_check"] = "on" if args.res_check else "off"
        build = subprocess.run(["bash", str(ROOT / "compile_all.sh"), args.category],
                               cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.out / "compile.log").write_text(build.stdout)
        build_rc = build.returncode
    coverage = json.loads((ROOT / "coverage.json").read_text())
    active = coverage["active"]
    if args.category != "all":
        active = [item for item in active if item["family"] == args.category]
    records = []
    for case in active:
        family, name = case["family"], case["name"]
        output_root = REPO / ("output/res_check" if args.res_check else "output")
        matches = list((output_root / f"microbenchmark/{family}/elf").rglob(f"{name}.elf"))
        if not matches:
            records.append({**case, "state": "COMPILE_FAIL", "elf": None})
            continue
        elf = matches[0]
        diss = Path(str(elf) + ".diss")
        opcode = expected_opcode(case)
        check_opcode = family != "scalar"
        diss_text = diss.read_text(errors="replace").upper() if diss.exists() else ""
        precise_match = False
        if family in ("memory", "cube", "fixp"):
            engine = "TLSU" if family == "memory" else "CUBE"
            precise_match = contains(
                diss_text, engine, canonical_operation(case["operation"]),
                canonical_dtype(case.get("dtype")),
            )
        else:
            precise_match = opcode in diss_text
        if not diss.exists() or (check_opcode and not precise_match):
            records.append({**case, "state": "DISS_MISSING_OPCODE", "elf": str(elf),
                            "diss": str(diss), "expected_opcode": opcode})
            continue
        if args.skip_run:
            records.append({**case, "state": "NOT_RUN", "elf": str(elf), "diss": str(diss)})
            continue
        try:
            command = [str(args.gfrun), "-t", "1"]
            if case.get("command_mnemonic") == "BSTART.TIMG2COL":
                command += ["-s", "softcore.multiThreadNum=4"]
            elif family == "fixp" and case.get("mode") in {
                    "shared", "s8_shared", "trans_a", "trans_b", "trans_ab"}:
                command += ["-s", "softcore.multiThreadNum=4"]
            command += ["-f", str(elf)]
            proc = subprocess.run(command,
                                  text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
            log = args.out / "logs" / f"{family}_{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(proc.stdout)
            ended = "Reach the End of Benchmark" in proc.stdout
            r2_ok = "R2 = 0" in proc.stdout
            if proc.returncode == 0 and ended and r2_ok:
                state = "PASS"
            elif args.res_check and ended and not r2_ok:
                state = "NUMERIC_FAIL"
            else:
                state = "RUN_FAIL"
            records.append({**case, "state": state, "elf": str(elf), "diss": str(diss),
                            "returncode": proc.returncode, "command": command,
                            "log": str(log)})
        except subprocess.TimeoutExpired:
            records.append({**case, "state": "TIMEOUT", "elf": str(elf), "diss": str(diss)})
    counts = Counter(item["state"] for item in records)
    payload = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "compiler_dir": str(args.compiler_dir), "gfrun": str(args.gfrun),
               "res_check": args.res_check,
               "build_returncode": build_rc, "counts": dict(counts), "results": records,
               "unsupported": coverage["unsupported"]}
    (args.out / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = ["# Microbenchmark report", "", "| State | Count |", "| --- | ---: |"]
    rows.extend(f"| {state} | {count} |" for state, count in sorted(counts.items()))
    rows += ["", "| Family | Testcase | State |", "| --- | --- | --- |"]
    rows.extend(f"| {r['family']} | {r['name']} | {r['state']} |" for r in records)
    (args.out / "result.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(dict(counts), sort_keys=True))
    bad = {"COMPILE_FAIL", "DISS_MISSING_OPCODE", "NUMERIC_FAIL", "RUN_FAIL", "TIMEOUT"}
    return 1 if build_rc or any(state in counts for state in bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
