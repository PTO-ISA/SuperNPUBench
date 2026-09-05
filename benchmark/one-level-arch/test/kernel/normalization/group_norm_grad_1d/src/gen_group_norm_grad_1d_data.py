#!/usr/bin/env python3
"""Generate group_norm_grad_1d host bins (HxW==1, pure Python, no numpy).

Bins written to --out-dir:
  tiling_info.bin  : 4 x int64 LE = (N, C, G, tile_d)
  dy.bin / x.bin   : N*C x float16
  mean.bin/rstd.bin: N*G x float32
  gamma.bin        : C x float16
  golden_dx.bin / golden_dgamma.bin / golden_dbeta.bin : float16

Math matches PyTorch GroupNorm1dBackward (fp32 accumulate, cast to fp16).
Default: N=512, C=64, G=8 (D=8), tile_d=8.
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
    / "kernel_normalization_group_norm_grad_1d_group_norm_grad_1d_DType__half_N512_C64_G8_PE4"
)


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


def pack_f16(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<H", f32_to_f16_bits(v)) for v in vals)


def pack_f32(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in vals)


def quantize_f16(vals: list[float]) -> list[float]:
    return [f16_bits_to_f32(f32_to_f16_bits(v)) for v in vals]


def randn(rng: random.Random, n: int, lo: float = -8.0, hi: float = 8.0) -> list[float]:
    out: list[float] = []
    for _ in range(n):
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        out.append(max(min(z, hi), lo))
    return out


def group_norm_grad_1d_ref(
    dy: list[float],
    x: list[float],
    mean: list[float],
    rstd: list[float],
    gamma: list[float],
    N: int,
    C: int,
    G: int,
) -> tuple[list[float], list[float], list[float]]:
    """fp32 reference matching CUDA GroupNorm1dBackward."""
    D = C // G
    s = 1.0 / D
    dx = [0.0] * (N * C)
    dgamma = [0.0] * C
    dbeta = [0.0] * C

    for n in range(N):
        for g in range(G):
            ng = n * G + g
            c0 = g * D
            m = mean[ng]
            r = rstd[ng]
            sum1 = 0.0
            sum2 = 0.0
            for i in range(D):
                c = c0 + i
                idx = n * C + c
                gv = gamma[c]
                dy_v = dy[idx]
                x_v = x[idx]
                sum1 += dy_v * x_v * gv
                sum2 += dy_v * gv
            c2 = (sum2 * m - sum1) * (r * r * r) * s
            c3 = -c2 * m - sum2 * r * s
            for i in range(D):
                c = c0 + i
                idx = n * C + c
                c1 = r * gamma[c]
                dx[idx] = c1 * dy[idx] + c2 * x[idx] + c3

    for n in range(N):
        for c in range(C):
            g = c // D
            ng = n * G + g
            idx = n * C + c
            dgamma[c] += dy[idx] * (x[idx] - mean[ng]) * rstd[ng]
            dbeta[c] += dy[idx]

    return dx, dgamma, dbeta


def gen_forward_stats(
    x: list[float], N: int, C: int, G: int, eps: float
) -> tuple[list[float], list[float]]:
    """Compute mean/rstd like GroupNorm forward (HxW=1)."""
    D = C // G
    mean: list[float] = []
    rstd: list[float] = []
    for n in range(N):
        for g in range(G):
            c0 = g * D
            acc = 0.0
            for i in range(D):
                acc += x[n * C + c0 + i]
            m = acc / D
            var = 0.0
            for i in range(D):
                d = x[n * C + c0 + i] - m
                var += d * d
            var /= D
            mean.append(m)
            rstd.append(1.0 / math.sqrt(var + eps))
    return mean, rstd


def gen_all(
    out_dir: Path,
    N: int,
    C: int,
    G: int,
    tile_d: int,
    eps: float,
    seed: int,
) -> None:
    assert C % G == 0, "C must be divisible by G"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    x = quantize_f16(randn(rng, N * C))
    dy = quantize_f16(randn(rng, N * C))
    gamma = quantize_f16(randn(rng, C, lo=-2.0, hi=2.0))
    mean, rstd = gen_forward_stats(x, N, C, G, eps)

    dx, dgamma, dbeta = group_norm_grad_1d_ref(dy, x, mean, rstd, gamma, N, C, G)

    (out_dir / "tiling_info.bin").write_bytes(struct.pack("<4q", N, C, G, tile_d))
    (out_dir / "dy.bin").write_bytes(pack_f16(dy))
    (out_dir / "x.bin").write_bytes(pack_f16(x))
    (out_dir / "mean.bin").write_bytes(pack_f32(mean))
    (out_dir / "rstd.bin").write_bytes(pack_f32(rstd))
    (out_dir / "gamma.bin").write_bytes(pack_f16(gamma))
    (out_dir / "golden_dx.bin").write_bytes(pack_f16(dx))
    (out_dir / "golden_dgamma.bin").write_bytes(pack_f16(dgamma))
    (out_dir / "golden_dbeta.bin").write_bytes(pack_f16(dbeta))

    print(f"wrote {out_dir}")
    print(f"  shape N={N} C={C} G={G} D={C // G} tile_d={tile_d}")
    print(f"  elems: X/dY/dX={N * C}, mean/rstd={N * G}, gamma={C}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--tile-d", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_CMP_DIR)
    parser.add_argument(
        "--also-src-data",
        action="store_true",
        help="Also write tiling_info.bin under src/data/ (optional)",
    )
    args = parser.parse_args()

    gen_all(args.out_dir, args.n, args.c, args.g, args.tile_d, args.eps, args.seed)
    if args.also_src_data:
        data_dir = SCRIPT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "tiling_info.bin").write_bytes(
            struct.pack("<4q", args.n, args.c, args.g, args.tile_d)
        )
        print(f"wrote {data_dir / 'tiling_info.bin'}")


if __name__ == "__main__":
    main()
