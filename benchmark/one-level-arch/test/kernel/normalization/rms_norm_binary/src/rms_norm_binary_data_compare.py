#!/usr/bin/env python3
"""Compare rms_norm_binary output.bin vs golden.bin (float16)."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CMP_DIR = (
    SCRIPT_DIR.parents[4]
    / "compare"
    / "kernel_normalization_rms_norm_binary_rms_norm_binary_DType__half"
)


def f16_bits_to_f32(h: int) -> float:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        val = 0.0 if mant == 0 else math.ldexp(mant / 1024.0, -14)
    elif exp == 31:
        val = math.nan if mant else math.inf
    else:
        val = math.ldexp(1.0 + mant / 1024.0, exp - 15)
    return -val if sign else val


def load_f16(path: Path) -> list[float]:
    data = path.read_bytes()
    n = len(data) // 2
    out = []
    for i in range(n):
        (h,) = struct.unpack_from("<H", data, i * 2)
        out.append(f16_bits_to_f32(h))
    return out


def compare(
    output_path: Path,
    golden_path: Path,
    *,
    atol: float,
    rtol: float,
    mse_tol: float,
) -> dict:
    out = load_f16(output_path)
    ref = load_f16(golden_path)
    if len(out) != len(ref):
        return {
            "status": "fail",
            "reason": f"size mismatch out={len(out)} golden={len(ref)}",
        }

    n = len(out)
    abs_diffs = []
    rel_diffs = []
    se = 0.0
    for a, b in zip(out, ref):
        d = abs(a - b)
        abs_diffs.append(d)
        se += (a - b) * (a - b)
        denom = max(abs(b), 1e-6)
        rel_diffs.append(d / denom)

    mse = se / n if n else 0.0
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    mean_abs = sum(abs_diffs) / n if n else 0.0
    max_rel = max(rel_diffs) if rel_diffs else 0.0
    within = all(
        (d <= atol) or (r <= rtol) for d, r in zip(abs_diffs, rel_diffs)
    )
    mse_ok = mse <= mse_tol
    status = "pass" if within and mse_ok else "fail"
    return {
        "status": status,
        "n": n,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mse": mse,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "mse_tol": mse_tol,
        "within_atol_or_rtol": within,
        "mse_ok": mse_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmp-dir", type=Path, default=DEFAULT_CMP_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--mse-tol", type=float, default=1e-3)
    args = parser.parse_args()

    output = args.output or (args.cmp_dir / "output.bin")
    golden = args.golden or (args.cmp_dir / "golden.bin")
    if not output.is_file():
        print(f"FAIL: missing {output}")
        return 2
    if not golden.is_file():
        print(f"FAIL: missing {golden}")
        return 2

    result = compare(
        output, golden, atol=args.atol, rtol=args.rtol, mse_tol=args.mse_tol
    )
    print(json.dumps(result, indent=2))
    print(
        f"=== rms_norm_binary precision: {result['status'].upper()} "
        f"(max_abs={result.get('max_abs')}, mse={result.get('mse')}) ==="
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
