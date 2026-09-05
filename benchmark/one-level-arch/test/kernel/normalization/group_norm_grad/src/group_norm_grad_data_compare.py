#!/usr/bin/env python3
"""Compare group_norm_grad outputs vs golden (float16, pure Python)."""

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
    / "kernel_normalization_group_norm_grad_group_norm_grad"
    "_DType__half_N32_C16_G8_HxW8192_PE4"
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


def compare_one(
    name: str,
    output_path: Path,
    golden_path: Path,
    *,
    atol: float,
    rtol: float,
    mse_tol: float,
) -> dict:
    if not output_path.is_file():
        return {"name": name, "status": "fail", "reason": f"missing {output_path}"}
    if not golden_path.is_file():
        return {"name": name, "status": "fail", "reason": f"missing {golden_path}"}

    out = load_f16(output_path)
    ref = load_f16(golden_path)
    if len(out) != len(ref):
        return {
            "name": name,
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
    within = all((d <= atol) or (r <= rtol) for d, r in zip(abs_diffs, rel_diffs))
    mse_ok = mse <= mse_tol
    status = "pass" if within and mse_ok else "fail"
    return {
        "name": name,
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
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--mse-tol", type=float, default=1e-3)
    args = parser.parse_args()

    results = []
    for name in ("dx", "dgamma", "dbeta"):
        results.append(
            compare_one(
                name,
                args.cmp_dir / f"{name}.bin",
                args.cmp_dir / f"golden_{name}.bin",
                atol=args.atol,
                rtol=args.rtol,
                mse_tol=args.mse_tol,
            )
        )

    print(json.dumps(results, indent=2))
    all_pass = all(r["status"] == "pass" for r in results)
    summary = ", ".join(
        f"{r['name']}={r['status']}"
        + (f"(max_abs={r.get('max_abs'):.4g})" if "max_abs" in r else "")
        for r in results
    )
    print(
        f"=== group_norm_grad precision: {'PASS' if all_pass else 'FAIL'} ({summary}) ==="
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
