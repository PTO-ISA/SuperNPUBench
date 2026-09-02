#!/usr/bin/env python3

import argparse
import math
import struct
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP16/BF16 QSMLA NPU output with FP32 CPU golden output."
    )
    parser.add_argument(
        "--actual", type=Path, default=HERE / "qsmla_onepass_npu_out.bin",
        help="NPU output in the format selected by --actual-dtype",
    )
    parser.add_argument(
        "--golden", type=Path, default=HERE / "qsmla_golden.bin",
        help="CPU golden output in little-endian FP32 format",
    )
    parser.add_argument("--b", type=int)
    parser.add_argument("--s1", type=int)
    parser.add_argument("--n1", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--actual-dtype", choices=("fp16", "bf16"), default="fp16",
        help="storage dtype of --actual (default: fp16)",
    )
    return parser.parse_args()


def read_values(path: Path, element_bytes: int, format_code: str):
    data = path.read_bytes()
    if len(data) % element_bytes != 0:
        raise ValueError(
            f"{path}: byte size {len(data)} is not divisible by {element_bytes}"
        )
    count = len(data) // element_bytes
    return struct.unpack(f"<{count}{format_code}", data)


def read_actual(path: Path, dtype: str):
    if dtype == "fp16":
        return read_values(path, 2, "e")
    words = read_values(path, 2, "H")
    return tuple(
        struct.unpack("<f", struct.pack("<I", word << 16))[0]
        for word in words
    )


def main() -> int:
    args = parse_args()
    shape_values = (args.b, args.s1, args.n1, args.d)
    if any(value is not None for value in shape_values) and not all(
        value is not None for value in shape_values
    ):
        print("--b, --s1, --n1 and --d must be specified together", file=sys.stderr)
        return 2
    if args.atol < 0 or args.rtol < 0:
        print("--atol and --rtol must be non-negative", file=sys.stderr)
        return 2

    try:
        actual = read_actual(args.actual, args.actual_dtype)
        golden = read_values(args.golden, 4, "f")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    if len(actual) != len(golden):
        print(
            f"element count mismatch: actual={len(actual)}, golden={len(golden)}",
            file=sys.stderr,
        )
        return 2

    if all(value is not None for value in shape_values):
        if any(value <= 0 for value in shape_values):
            print("BSND dimensions must be positive", file=sys.stderr)
            return 2
        expected = args.b * args.s1 * args.n1 * args.d
        if expected != len(actual):
            print(
                f"shape element count mismatch: BSND[{args.b},{args.s1},{args.n1},{args.d}] "
                f"expects {expected}, files contain {len(actual)}",
                file=sys.stderr,
            )
            return 2

    errors = [abs(float(a) - float(g)) for a, g in zip(actual, golden)]
    passed = [
        math.isfinite(float(a))
        and math.isfinite(float(g))
        and error <= args.atol + args.rtol * abs(float(g))
        for a, g, error in zip(actual, golden, errors)
    ]
    passed_count = sum(passed)
    count = len(actual)

    print(f"actual  = {args.actual}")
    print(f"golden  = {args.golden}")
    if all(value is not None for value in shape_values):
        print(f"shape   = BSND[{args.b},{args.s1},{args.n1},{args.d}]")
    print(f"tol     = atol={args.atol:g}, rtol={args.rtol:g}")
    print(f"passed  = {passed_count}/{count} ({100 * passed_count / count:.6f}%)" if count else "passed  = 0/0")
    print(f"failed  = {count - passed_count}")
    print(f"max_abs = {max(errors):.9f}" if errors else "max_abs = 0.000000000")
    print(f"mean_abs= {sum(errors) / count:.9f}" if count else "mean_abs= 0.000000000")
    print(f"nan_npu = {sum(math.isnan(float(value)) for value in actual)}")
    print(f"nan_ref = {sum(math.isnan(float(value)) for value in golden)}")
    return 0 if passed_count == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
