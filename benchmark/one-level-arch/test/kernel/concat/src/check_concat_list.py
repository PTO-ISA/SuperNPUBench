#!/usr/bin/env python3
"""Runtime tensor-list/axis validation: one ELF per dtype, many input shapes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from check_concat_two import CASE_DIR, ROOT, DTYPES, random_bits

MAX_INPUTS, INPUT_SIZE, OUTPUT_SIZE, GUARD = 64, 4096, 64 * 4096, 32


def cases():
    result = []
    def add(name, shapes, axis, **kw):
        result.append(dict(name=name, shapes=shapes, axis=axis, status=0, **kw))
    add("three_inputs", [(3, 5), (3, 7), (3, 2)], 1)
    add("one_input", [(3, 129)], -1)
    add("tail_boundaries", [(5, 127), (5, 128), (5, 129)], -1)
    add("axis0", [(1, 5), (3, 5), (7, 5)], 0)
    add("many33", [(4, 129)] * 33, 1)
    add("many64", [(1,), (2,)] * 32, 0)
    for axis in range(4):
        shapes = []
        for width in [1, 3, 7]:
            shape = [2, 3, 5, 7]
            shape[axis] = width
            shapes.append(tuple(shape))
        add(f"rank4_axis{axis}", shapes, axis)
        add(f"rank4_negative{axis}", shapes, axis - 4)
    add("rank8", [(2, 1, 1, 3, 1, 1, 1, 5), (2, 1, 1, 7, 1, 1, 1, 5)], 3)
    add("empty_members", [(0,), (3, 0), (3, 5), (3, 0), (3, 7)], 1)
    add("empty_outer", [(0, 3), (0, 7)], 1, null_inputs=True, flags=2)
    add("empty_inner", [(2, 3, 0), (2, 7, 0)], 1, null_inputs=True, flags=2)
    add("all_empty", [(0,), (0,)], 99, flags=2)
    add("axis_empty", [(2, 0), (2, 0)], 1, flags=2)
    add("empty_list", [], 99, flags=2)
    # The caller-provided status oracle is independent of kernel logic.
    invalid = [
        ("null_list", [(2, 3)], 1, 1, {"flags": 1}),
        ("rank_zero", [()], 0, 2, {}),
        ("rank_nine", [(1,) * 9], 0, 2, {}),
        ("rank_mismatch", [(2, 3), (2, 3, 1)], 1, 2, {}),
        ("axis_high", [(2, 3)], 2, 3, {}),
        ("axis_low", [(2, 3)], -3, 3, {}),
        ("shape_mismatch", [(2, 3), (3, 4)], 1, 4, {}),
        ("negative_dimension", [(2, -3)], 1, 4, {}),
        ("null_input", [(2, 3)], 1, 5, {"null_inputs": True}),
        ("null_output", [(2, 3)], 1, 5, {"flags": 2}),
        ("short_input", [(2, 3)], 1, 6, {"input_capacity": 5}),
        ("short_output", [(2, 3), (2, 4)], 1, 6, {"output_capacity": 13}),
        ("axis_sum_overflow", [(2**63 - 1,), (1,)], 0, 7, {}),
        ("product_overflow", [(2**62, 8)], 1, 7, {}),
        ("byte_size_overflow", [(2**62,)], 0, 7, {}),
        ("output_overlap", [(2, 3)], 1, 8, {"alias": True}),
        ("late_invalid_input", [(2, 3), (2, 4), (3, 5)], 1, 4, {}),
    ]
    for name, shapes, axis, status, kw in invalid:
        result.append(dict(name=name, shapes=shapes, axis=axis, status=status, **kw))
    return result


def run_case(case, cpp, dtype, elf, args, directory, index):
    logdir = directory / case["name"]
    logdir.mkdir()
    compare = ROOT / "compare" / elf.stem
    compare.mkdir(parents=True, exist_ok=True)
    shape_list = case["shapes"]
    metadata = np.zeros(4 + MAX_INPUTS * 11, dtype="<i8")
    metadata[:4] = [len(shape_list), case["axis"], case.get("output_capacity", OUTPUT_SIZE), case.get("flags", 0)]
    rng = np.random.default_rng(args.seed + index)
    bank = random_bits(rng, (len(shape_list), INPUT_SIZE), dtype)
    for i, shape in enumerate(shape_list):
        start = 4 + i * 11
        metadata[start] = len(shape)
        metadata[start + 1:start + 1 + min(len(shape), 8)] = shape[:8]
        metadata[start + 9] = case.get("input_capacity", INPUT_SIZE)
        metadata[start + 10] = 2 if case.get("alias") else int(case.get("null_inputs", False))
    metadata.tofile(compare / "metadata.bin")
    bank.tofile(compare / "inputs.bin")
    initial = bytes([0xA5]) * ((OUTPUT_SIZE + 2 * GUARD) * dtype.itemsize)
    (compare / "output.bin").write_bytes(initial)
    (compare / "status.bin").write_bytes(bytes([0xFF]) * (4 * 10 * 8))
    expected = initial
    expected_shape = None
    if case["status"] == 0:
        arrays = []
        for i, shape in enumerate(shape_list):
            if shape == (0,):
                continue  # A5 filters 1-D empty tensors before rank/axis checks.
            assert math.prod(shape) <= INPUT_SIZE
            arrays.append(bank[i, :math.prod(shape)].reshape(shape))
        golden = np.concatenate(arrays, axis=case["axis"]) if arrays else np.empty((0,), dtype=dtype)
        expected_shape = golden.shape
        payload = golden.tobytes()
        assert len(payload) <= OUTPUT_SIZE * dtype.itemsize
        begin = GUARD * dtype.itemsize
        expected = initial[:begin] + payload + initial[begin + len(payload):]
    command = [str(args.gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    with (logdir / "gfrun.log").open("w") as log:
        process = subprocess.run(command, cwd=args.gfrun.parent.parent, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=args.timeout)
    raw = (logdir / "gfrun.log").read_text(errors="replace")
    status = np.fromfile(compare / "status.bin", dtype="<i8").reshape(4, 10)
    actual = (compare / "output.bin").read_bytes()
    checks = {
        "exit_zero": process.returncode == 0,
        "benchmark_end": "Reach the End of Benchmark" in raw,
        "r2_zero": bool(re.search(r"\bR2\s*=\s*0\b", raw)) and not re.search(r"\bR2\s*=\s*[1-9]", raw),
        "workers": "Starting PE1..PE3 from lightweight worker entry" in raw,
        "all_pe_status": bool(np.all(status[:, 0] == case["status"])),
        "exact_output_and_guards": actual == expected,
    }
    if expected_shape is not None:
        checks["inferred_shape"] = bool(np.all(status[:, 1] == len(expected_shape)) and
                                        np.all(status[:, 2:2 + len(expected_shape)] == expected_shape))
    else:
        checks["metadata_unchanged_on_error"] = bool(np.all(status[:, 1] == 99) and np.all(status[:, 2:] == -99))
    result = {**case, "dtype": cpp, "checks": checks, "statuses": status[:, 0].tolist(),
              "pass": all(checks.values())}
    (logdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    # Preserve runtime metadata for reproducing failures without recompilation.
    (logdir / "metadata.bin").write_bytes(metadata.tobytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", default=os.environ.get("COMPILER_DIR"))
    parser.add_argument("--gfrun", default=os.environ.get("GFRUN") or shutil.which("gfrun"))
    parser.add_argument("--artifacts", type=Path, default=ROOT / "output/kernel/concat/validation")
    parser.add_argument("--case", action="append", choices=[c["name"] for c in cases()])
    parser.add_argument("--dtype", action="append", choices=list(DTYPES))
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not args.compiler_dir or not args.gfrun:
        parser.error("set COMPILER_DIR and GFRUN")
    args.compiler_dir = Path(args.compiler_dir).resolve()
    args.gfrun = Path(args.gfrun).resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="concat-list-", dir=args.artifacts.resolve()))
    results, identities = [], {}
    for name in args.dtype or list(DTYPES):
        cpp, dtype = DTYPES[name]
        subdir = directory / name
        subdir.mkdir()
        with (subdir / "compile.log").open("w") as log:
            build = subprocess.run(["make", "-B", "-j1", "TESTCASE=concat_list", f"DTYPE={cpp}",
                                    f"COMPILER_DIR={args.compiler_dir}"], cwd=CASE_DIR, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=args.timeout)
        if build.returncode:
            results.append({"dtype": name, "pass": False, "error": "compile failed"})
            print(f"{name}: COMPILE FAIL {subdir}", flush=True)
            continue
        elf = ROOT / "output/kernel/concat/elf" / f"kernel_concat_concat_list_{cpp}_PE4.elf"
        identities[name] = hashlib.sha256(elf.read_bytes()).hexdigest()
        for index, case in enumerate(cases()):
            if args.case and case["name"] not in args.case:
                continue
            # Byte-size overflow is dtype-dependent: use a size that overflows
            # PTRDIFF_MAX for every tested element size.
            if case["name"] == "byte_size_overflow":
                case = {**case, "shapes": [(2**63 - 1,), (0,)]}
                if dtype.itemsize == 1:
                    case = {**case, "shapes": [(2**62, 4)]}
            try:
                result = run_case(case, cpp, dtype, elf, args, subdir, index)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                result = {"name": case["name"], "dtype": name, "pass": False, "error": str(error)}
            results.append(result)
            print(f"{name}/{case['name']}: {'PASS' if result['pass'] else 'FAIL'}", flush=True)
        assert hashlib.sha256(elf.read_bytes()).hexdigest() == identities[name]
    summary = {"pass": sum(r["pass"] for r in results), "fail": sum(not r["pass"] for r in results),
               "elf_sha256": identities, "seed": args.seed, "cases": results}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS={summary['pass']} FAIL={summary['fail']} artifacts={directory}")
    return int(summary["fail"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
