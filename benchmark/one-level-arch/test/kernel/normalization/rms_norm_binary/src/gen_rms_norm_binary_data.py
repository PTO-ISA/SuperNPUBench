#!/usr/bin/env python3
"""Generate rms_norm_binary host bins for [16, 16384] fp16.

tiling_info.bin  : 5 x int64 LE = (g_a, g_r, tile_a, tile_r, pow_r)
input.bin        : g_a * g_r x float16
golden.bin       : out = x * rsqrt(mean(x^2)+eps)  (fp32 compute → fp16)

pow_r: power of two with pow_r < g_r <= 2 * pow_r (Pass1 split).
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CMP_DIR = (
    SCRIPT_DIR.parents[4]
    / "compare"
    / "kernel_normalization_rms_norm_binary_rms_norm_binary_DType__half_gA16_gR16384_PE4"
)
DATA_DIR = SCRIPT_DIR / "data"


def f32_to_f16_bits(x: float) -> int:
    b = struct.pack("<f", float(x))
    f32 = struct.unpack("<I", b)[0]
    sign = (f32 >> 16) & 0x8000
    exp = (f32 >> 23) & 0xFF
    mant = f32 & 0x7FFFFF
    if exp == 255:
        return sign | 0x7C00 | (0x200 if mant else 0)
    new_exp = exp - 127 + 15
    if new_exp >= 31:
        return sign | 0x7C00
    if new_exp <= 0:
        if new_exp < -10:
            return sign
        mant |= 0x800000
        shift = 14 - new_exp
        half = mant >> shift
        if (mant >> (shift - 1)) & 1:
            half += 1
        return sign | half
    half = (new_exp << 10) | (mant >> 13)
    round_bit = (mant >> 12) & 1
    sticky = mant & 0xFFF
    if round_bit and (sticky or (half & 1)):
        half += 1
    return sign | half


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


def pack_f16_list(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<H", f32_to_f16_bits(v)) for v in vals)


def rms_norm_rows(x_f16: list[float], g_a: int, g_r: int, eps: float) -> list[float]:
    y: list[float] = []
    for ia in range(g_a):
        row = x_f16[ia * g_r : (ia + 1) * g_r]
        mean_sq = sum(v * v for v in row) / g_r
        inv_rms = 1.0 / math.sqrt(mean_sq + eps)
        y.extend(v * inv_rms for v in row)
    return y


def write_tiling_info(
    path: Path, g_a: int, g_r: int, tile_a: int, tile_r: int, pow_r: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<5q", g_a, g_r, tile_a, tile_r, pow_r))
    print(f"wrote {path}  tiling=({g_a},{g_r},{tile_a},{tile_r},{pow_r})")


def gen_all(
    out_dir: Path,
    g_a: int,
    g_r: int,
    tile_a: int,
    tile_r: int,
    pow_r: int,
    eps: float,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    x_f32: list[float] = []
    for _ in range(g_a * g_r):
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        x_f32.append(max(min(z, 8.0), -8.0))

    x_f16 = [f16_bits_to_f32(f32_to_f16_bits(v)) for v in x_f32]
    y_f32 = rms_norm_rows(x_f16, g_a, g_r, eps)

    write_tiling_info(out_dir / "tiling_info.bin", g_a, g_r, tile_a, tile_r, pow_r)
    in_bytes = pack_f16_list(x_f16)
    gold_bytes = pack_f16_list(y_f32)
    (out_dir / "input.bin").write_bytes(in_bytes)
    (out_dir / "golden.bin").write_bytes(gold_bytes)
    print(f"wrote {out_dir / 'input.bin'}  elems={g_a * g_r} bytes={len(in_bytes)}")
    print(f"wrote {out_dir / 'golden.bin'} elems={g_a * g_r} bytes={len(gold_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g-a", type=int, default=16)
    parser.add_argument("--g-r", type=int, default=16384)
    parser.add_argument("--tile-a", type=int, default=1)
    parser.add_argument("--tile-r", type=int, default=8192)
    parser.add_argument("--pow-r", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_CMP_DIR)
    parser.add_argument("--also-src-data", action="store_true")
    args = parser.parse_args()

    gen_all(
        args.out_dir,
        args.g_a,
        args.g_r,
        args.tile_a,
        args.tile_r,
        args.pow_r,
        args.eps,
        args.seed,
    )
    if args.also_src_data:
        write_tiling_info(
            DATA_DIR / "tiling_info.bin",
            args.g_a,
            args.g_r,
            args.tile_a,
            args.tile_r,
            args.pow_r,
        )


if __name__ == "__main__":
    main()
