#!/usr/bin/env python3
"""Generate group_norm_grad host bins (HxW>1, pure Python, no numpy).

Bins:
  tiling_info.bin : 5 x int64 LE = (N, C, G, HxW, tile_hw)
  dy.bin / x.bin  : N*C*HxW x float16
  mean/rstd.bin   : N*G x float32
  gamma.bin       : C x float16
  golden_dx / golden_dgamma / golden_dbeta : float16

Math matches PyTorch GroupNormBackward (HxW>1): spatial ds/db then fused c2/c3.
Default: N=32, C=16, G=8, HxW=8192 (D=2), tile_hw=8192.
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
    / "kernel_normalization_group_norm_grad_group_norm_grad"
    "_DType__half_N32_C16_G8_HxW8192_PE4"
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


def idx(n: int, c: int, hw: int, C: int, HxW: int) -> int:
    return (n * C + c) * HxW + hw


def gen_forward_stats(
    x: list[float], N: int, C: int, G: int, HxW: int, eps: float
) -> tuple[list[float], list[float]]:
    D = C // G
    mean: list[float] = []
    rstd: list[float] = []
    denom = D * HxW
    for n in range(N):
        for g in range(G):
            c0 = g * D
            acc = 0.0
            for i in range(D):
                for hw in range(HxW):
                    acc += x[idx(n, c0 + i, hw, C, HxW)]
            m = acc / denom
            var = 0.0
            for i in range(D):
                for hw in range(HxW):
                    d = x[idx(n, c0 + i, hw, C, HxW)] - m
                    var += d * d
            var /= denom
            mean.append(m)
            rstd.append(1.0 / math.sqrt(var + eps))
    return mean, rstd


def group_norm_grad_ref(
    dy: list[float],
    x: list[float],
    mean: list[float],
    rstd: list[float],
    gamma: list[float],
    N: int,
    C: int,
    G: int,
    HxW: int,
) -> tuple[list[float], list[float], list[float]]:
    """fp32 reference matching CUDA GroupNormBackward HxW>1."""
    D = C // G
    s = 1.0 / (D * HxW)
    ds = [0.0] * (N * C)
    db = [0.0] * (N * C)

    for n in range(N):
        for c in range(C):
            nc = n * C + c
            for hw in range(HxW):
                i = idx(n, c, hw, C, HxW)
                ds[nc] += dy[i] * x[i]
                db[nc] += dy[i]

    dx = [0.0] * (N * C * HxW)
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
                nc = n * C + c
                gv = gamma[c]
                sum1 += ds[nc] * gv
                sum2 += db[nc] * gv
            c2 = (sum2 * m - sum1) * (r * r * r) * s
            c3 = -c2 * m - sum2 * r * s
            for i in range(D):
                c = c0 + i
                c1 = r * gamma[c]
                for hw in range(HxW):
                    j = idx(n, c, hw, C, HxW)
                    dx[j] = c1 * dy[j] + c2 * x[j] + c3

    dgamma = [0.0] * C
    dbeta = [0.0] * C
    for n in range(N):
        for c in range(C):
            g = c // D
            ng = n * G + g
            nc = n * C + c
            dgamma[c] += (ds[nc] - db[nc] * mean[ng]) * rstd[ng]
            dbeta[c] += db[nc]

    return dx, dgamma, dbeta


def gen_all(
    out_dir: Path,
    N: int,
    C: int,
    G: int,
    HxW: int,
    tile_hw: int,
    eps: float,
    seed: int,
) -> None:
    assert C % G == 0, "C must be divisible by G"
    assert HxW > 1, "use group_norm_grad_1d for HxW==1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    numel = N * C * HxW
    x = quantize_f16(randn(rng, numel))
    dy = quantize_f16(randn(rng, numel))
    gamma = quantize_f16(randn(rng, C, lo=-2.0, hi=2.0))
    mean, rstd = gen_forward_stats(x, N, C, G, HxW, eps)

    dx, dgamma, dbeta = group_norm_grad_ref(
        dy, x, mean, rstd, gamma, N, C, G, HxW
    )

    (out_dir / "tiling_info.bin").write_bytes(
        struct.pack("<5q", N, C, G, HxW, tile_hw)
    )
    (out_dir / "dy.bin").write_bytes(pack_f16(dy))
    (out_dir / "x.bin").write_bytes(pack_f16(x))
    (out_dir / "mean.bin").write_bytes(pack_f32(mean))
    (out_dir / "rstd.bin").write_bytes(pack_f32(rstd))
    (out_dir / "gamma.bin").write_bytes(pack_f16(gamma))
    (out_dir / "golden_dx.bin").write_bytes(pack_f16(dx))
    (out_dir / "golden_dgamma.bin").write_bytes(pack_f16(dgamma))
    (out_dir / "golden_dbeta.bin").write_bytes(pack_f16(dbeta))

    print(f"wrote {out_dir}")
    print(f"  shape N={N} C={C} G={G} HxW={HxW} D={C // G}")
    print(f"  elems={numel}, mean/rstd={N * G}, gamma={C}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--c", type=int, default=16)
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--hxw", type=int, default=8192)
    parser.add_argument("--tile-hw", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_CMP_DIR)
    parser.add_argument("--also-src-data", action="store_true")
    args = parser.parse_args()

    gen_all(
        args.out_dir,
        args.n,
        args.c,
        args.g,
        args.hxw,
        args.tile_hw,
        args.eps,
        args.seed,
    )
    if args.also_src_data:
        data_dir = SCRIPT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "tiling_info.bin").write_bytes(
            struct.pack("<5q", args.n, args.c, args.g, args.hxw, args.tile_hw)
        )
        print(f"wrote {data_dir / 'tiling_info.bin'}")


if __name__ == "__main__":
    main()
