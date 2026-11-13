#!/usr/bin/env python3
"""Compile two-input concat and compare gfrun output bit-for-bit with NumPy."""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
DTYPES = {
    "float32": ("float", np.dtype("<f4")),
    "float16": ("__half", np.dtype("<f2")),
    "int32": ("int32_t", np.dtype("<i4")),
    "uint8": ("uint8_t", np.dtype("u1")),
}
# name, left shape, right shape, axis, dtype, tile elements
CASES = [
    ("aligned", (4, 128), (4, 128), 1, "float32", 128),
    ("axis0", (3, 5), (2, 5), 0, "float32", 128),
    ("negative_axis", (3, 5), (3, 7), -1, "float32", 128),
    ("middle_axis", (2, 3, 5, 7), (2, 4, 5, 7), 1, "float32", 128),
    ("multi_tile_tail", (5, 257), (5, 131), 1, "float32", 128),
    ("one_dim", (513,), (259,), 0, "int32", 128),
    ("large_offsets", (129, 131), (129, 257), 1, "int32", 128),
    ("fp16_tail", (7, 133), (7, 9), 1, "float16", 128),
    ("byte_tail", (5, 259), (5, 3), 1, "uint8", 128),
    ("small_tile", (5, 35), (5, 67), 1, "float32", 32),
    ("tiny", (1,), (1,), 0, "float32", 128),
    ("empty_left", (3, 0, 5), (3, 7, 5), 1, "float32", 128),
    ("empty_right", (3, 7), (3, 0), 1, "int32", 128),
    ("both_empty", (2, 0), (2, 0), 1, "float32", 128),
]


def random_bits(rng, shape, dtype):
    # Copy must preserve arbitrary bits, including NaNs and signed zero.
    raw = rng.integers(0, 256, math.prod(shape) * dtype.itemsize, dtype=np.uint8)
    if dtype.kind == "f" and raw.size >= 4 * dtype.itemsize:
        bits = ([0, 0x80000000, 0x7FC01234, 0x7F800000] if dtype.itemsize == 4
                else [0, 0x8000, 0x7E12, 0x7C00])
        raw[:4 * dtype.itemsize] = np.array(bits, dtype=f"<u{dtype.itemsize}").view(np.uint8)
    return raw.view(dtype).reshape(shape)


def run_case(case, args, artifacts, rng):
    name, left_shape, right_shape, axis, type_name, tile = case
    dtype_cpp, dtype = DTYPES[type_name]
    axis %= len(left_shape)
    outer = math.prod(left_shape[:axis])
    inner = math.prod(left_shape[axis + 1:])
    left_axis, right_axis = left_shape[axis], right_shape[axis]
    stem = (f"kernel_concat_concat_two_O{outer}_L{left_axis}_R{right_axis}"
            f"_I{inner}_T{tile}_{dtype_cpp}_PE4")
    elf = ROOT / "output/kernel/concat/elf" / f"{stem}.elf"
    logs = artifacts / name
    logs.mkdir()
    build_cmd = ["make", "-B", "-j1", "TESTCASE=concat_two", "res_check=on",
                 f"COMPILER_DIR={args.compiler_dir}", f"OUTER={outer}",
                 f"LEFT={left_axis}", f"RIGHT={right_axis}", f"INNER={inner}",
                 f"TILE={tile}", f"DTYPE={dtype_cpp}"]
    result = {"name": name, "left_shape": left_shape, "right_shape": right_shape,
              "axis": axis, "dtype": type_name, "elf": str(elf), "status": "FAIL"}
    with (logs / "compile.log").open("w") as log:
        built = subprocess.run(build_cmd, cwd=CASE_DIR, stdout=log,
                               stderr=subprocess.STDOUT, timeout=args.timeout)
    result["compile_rc"] = built.returncode
    if built.returncode:
        return result

    compare = ROOT / "compare" / stem
    compare.mkdir(parents=True, exist_ok=True)
    left = random_bits(rng, left_shape, dtype)
    right = random_bits(rng, right_shape, dtype)
    golden = np.concatenate((left, right), axis=axis)
    expected = golden.tobytes(order="C")
    guard = bytes([0xA5]) * (32 * dtype.itemsize)
    initial = guard + bytes([0xCD]) * len(expected) + guard
    left.tofile(compare / "left.bin")
    right.tofile(compare / "right.bin")
    (compare / "golden.bin").write_bytes(expected)
    (compare / "output.bin").write_bytes(initial)
    command = [str(args.gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    with (logs / "gfrun.log").open("w") as log:
        run = subprocess.run(command, cwd=args.gfrun.parent.parent, stdout=log,
                             stderr=subprocess.STDOUT, timeout=args.timeout)
    output = (logs / "gfrun.log").read_text(errors="replace")
    actual = (compare / "output.bin").read_bytes()
    payload = actual[len(guard):len(guard) + len(expected)]
    checks = {
        "exit_zero": run.returncode == 0,
        "benchmark_end": "Reach the End of Benchmark" in output,
        "r2_zero": bool(re.search(r"\bR2\s*=\s*0\b", output))
                   and not re.search(r"\bR2\s*=\s*[1-9]", output),
        "workers_started": "Starting PE1..PE3 from lightweight worker entry" in output,
        "exact_size": len(actual) == len(initial),
        "guards_intact": actual[:len(guard)] == guard and actual[-len(guard):] == guard,
        "bitwise_equal": payload == expected,
    }
    result.update(run_rc=run.returncode, checks=checks, elements=golden.size,
                  bytes=len(expected), status="PASS" if all(checks.values()) else "FAIL")
    if len(payload) == len(expected):
        result["mismatched_bytes"] = int(np.count_nonzero(
            np.frombuffer(payload, dtype=np.uint8) != np.frombuffer(expected, dtype=np.uint8)))
    (logs / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-dir", default=os.environ.get("COMPILER_DIR"))
    parser.add_argument("--gfrun", default=os.environ.get("GFRUN") or shutil.which("gfrun"))
    parser.add_argument("--artifacts", type=Path,
                        default=Path(os.environ.get("ARTIFACTS", ROOT / "output/kernel/concat/validation")))
    parser.add_argument("--case", action="append", choices=[c[0] for c in CASES])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if not args.compiler_dir or not args.gfrun:
        parser.error("set COMPILER_DIR and GFRUN or pass --compiler-dir and --gfrun")
    args.compiler_dir = Path(args.compiler_dir).expanduser().resolve()
    args.gfrun = Path(args.gfrun).expanduser().resolve()
    if not (args.compiler_dir / "clang++").is_file() or not args.gfrun.is_file():
        parser.error("compiler or gfrun executable does not exist")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="concat-two-", dir=args.artifacts.resolve()))
    rng = np.random.default_rng(args.seed)
    results = []
    for case in CASES:
        if args.case and case[0] not in args.case:
            continue
        try:
            result = run_case(case, args, artifacts, rng)
        except (OSError, subprocess.TimeoutExpired) as error:
            result = {"name": case[0], "status": "FAIL", "error": str(error)}
        results.append(result)
        print(f"{result['name']}: {result['status']}", flush=True)
    report = {"time": datetime.datetime.now().isoformat(), "seed": args.seed,
              "compiler_dir": str(args.compiler_dir), "gfrun": str(args.gfrun),
              "cases": results, "pass": sum(r["status"] == "PASS" for r in results),
              "fail": sum(r["status"] != "PASS" for r in results)}
    (artifacts / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS={report['pass']} FAIL={report['fail']} artifacts={artifacts}")
    return int(report["fail"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
