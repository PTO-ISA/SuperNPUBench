#!/usr/bin/env python3
"""Generate DynamicHiF4Quant host bins: input / golden output / golden scale.

Faithful Python port of dynamic_hi_f4_quant_tail.h, matching the PTO-ISA kernel
op-for-op (bf16 arithmetic with round-to-nearest-even, mirroring TMULS/TMUL/
TRECIP/TROWEXPANDMUL and the TCVT->hif4 lane quantizer). Authoritative algorithm
= DESIGN.md §2.1 + pto-spec asl/arch/data-types/formats/{e6m2,rcpe6m2,hif4-scale,
hif4x2}.asl:

  per 64-block, per row:
    Vmax16[g] = max |x| over lanes {4g..4g+3}      (g=0..15)
    Vmax8 [g] = max(Vmax16[2g],Vmax16[2g+1])        (g=0..7)  = max over 8 lanes
    Vmax      = max over all 64
    SF        = bf16(Vmax * bf16(1/7))
    e6m2      = round_half_up(SF to 2 mantissa bits)  (kernel quantize_e6m2)
                q_bf16 = E6M2FiniteValue exactly; e6m2_byte = ((q_bits>>5)-316)&0xFF
    rec       = bf16(1 / q_bf16)                       (TRECIP; = 1/E6M2FiniteValue)
    E1_8 [g]  = ( bf16(Vmax8[g]*rec) >= 4 ) ? 1 : 0                     (g=0..7)
    E1_16[g]  = ( bf16(bf16(Vmax16[g]*rec)*2^-E1_8[g//2]) >= 2 ) ? 1:0  (g=0..15)
    S[j]      = E6M2FiniteValue * 2^(E1_8[j//8] + E1_16[j//4])   (HiF4ScaleFiniteValue)
    y[j]      = hif4_lane_quant( x[j] / S[j] )     (kernel: x * rec*2^-E1_8*2^-E1_16)
    scale_word (U32) = e6m2_byte | (sum E1_8[g]<<(8+g)) | (sum E1_16[g]<<(16+g))

Output files (per ELF compare dir):
  input.bin        : M*N x bf16 (2 bytes each), row-major
  golden.bin       : M*(N/2) bytes, fp4 packed 2/byte (low nibble = even element)
  scale_golden.bin : M*numKb x uint32 (4 bytes each), row-major (row, kb)
"""

import argparse
import math
import random
import struct
from pathlib import Path

import numpy as np  # float32 intermediates mirror the model (bf16->f32 op ->bf16 RNE)

BLOCK_SIZE = 64
BF16_INV7 = 0x3E12  # bf16 bits for 1/7 (kernel HIF4_INV7_B)


# --- bf16 domain --------------------------------------------------------------
def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even f32 -> bf16 (matches hardware bf16 rounding)."""
    u = f32_bits(x)
    if (u & 0x7F800000) == 0x7F800000:  # inf / nan: keep top 16
        return (u >> 16) & 0xFFFF
    bias = 0x7FFF + ((u >> 16) & 1)
    return ((u + bias) >> 16) & 0xFFFF


def bf16_bits_f32(b: int) -> float:
    return bits_f32((b & 0xFFFF) << 16)


def bf16_val(x: float) -> float:
    """Round a real to the nearest bf16 value (RNE), return as python float."""
    return bf16_bits_f32(f32_to_bf16_bits(x))


def bf16_mul(a: float, b: float) -> float:
    """bf16 * bf16 -> bf16 (RNE). Model路径: bf16->f32, f32 乘, ->bf16 RNE。用 float32
    中间量精确镜像(否则 double 中间量偶有 1-ULP 差致近阈值 E1 翻转)。"""
    return bf16_bits_f32(f32_to_bf16_bits(float(np.float32(a) * np.float32(b))))


def bf16_recip(x: float) -> float:
    """TRECIP(bf16): 模型 = bf16->f32, (float)1.0/x, ->bf16 RNE (Frecip<float>)。"""
    return bf16_bits_f32(f32_to_bf16_bits(float(np.float32(1.0) / np.float32(x))))


# --- e6m2 quantize (kernel quantize_e6m2, round-half-up @ bit4) ---------------
def quantize_e6m2(sf_val: float):
    sf_bits = f32_to_bf16_bits(sf_val)              # sf already bf16 -> idempotent
    q_bits = (sf_bits + 0x10) & 0xFFE0              # round-half-up, keep mant [6:5]
    q_val = bf16_bits_f32(q_bits)                   # = E6M2FiniteValue (exact bf16)
    e6m2_byte = ((q_bits >> 5) - 316) & 0xFF        # ((exp8-79)<<2)|m2
    return q_val, e6m2_byte


def e6m2_byte_value(byte: int) -> float:
    """Authoritative E6M2FiniteValue (e6m2.asl)."""
    if byte == 0xFF:
        return float("nan")
    exp6 = (byte >> 2) & 0x3F
    m2 = byte & 0x3
    return (1.0 + m2 / 4.0) * (2.0 ** (exp6 - 48))


# --- hif4 lane quantizer (hif4x2.asl lane table {0,.25,..,1.75}, signed) ------
def hif4_lane_quant(v: float) -> int:
    """Round v to nearest hif4 lane; return 4-bit nibble (sign<<3 | code)."""
    if v != v:  # NaN
        return 0
    sign = 1 if (v < 0.0 or (v == 0.0 and math.copysign(1.0, v) < 0.0)) else 0
    av = abs(v)
    # code = round(av/0.25), round-half-to-even, clamp [0,7] (saturate at 1.75)
    scaled = av / 0.25
    fl = math.floor(scaled)
    frac = scaled - fl
    if frac < 0.5:
        code = fl
    elif frac > 0.5:
        code = fl + 1
    else:  # exact tie -> even
        code = fl if (fl % 2 == 0) else fl + 1
    code = max(0, min(7, code))
    return (sign << 3) | code


HIF4_LANE_MAG = [0.25 * c for c in range(8)]  # code -> magnitude


# --- golden encode ------------------------------------------------------------
def encode_row_block(xblk):
    """xblk: 64 bf16 python-float values. Returns (nibbles[64], scale_word)."""
    ax = [abs(v) for v in xblk]
    vmax16 = [max(ax[4 * g:4 * g + 4]) for g in range(16)]
    vmax8 = [max(vmax16[2 * g], vmax16[2 * g + 1]) for g in range(8)]
    vmax = max(vmax8)

    sf = bf16_mul(vmax, bf16_bits_f32(BF16_INV7))
    q_val, e6m2_byte = quantize_e6m2(sf)
    rec = bf16_recip(q_val) if q_val != 0.0 else 0.0

    e1_8 = [0] * 8
    for g in range(8):
        t8 = bf16_mul(vmax8[g], rec)
        e1_8[g] = 1 if t8 >= 4.0 else 0
    e1_16 = [0] * 16
    for g in range(16):
        t16 = bf16_mul(vmax16[g], rec)
        f8 = 0.5 if e1_8[g // 2] else 1.0
        t16 = bf16_mul(t16, f8)
        e1_16[g] = 1 if t16 >= 2.0 else 0

    nibbles = [0] * 64
    for g in range(16):
        f8 = 0.5 if e1_8[g // 2] else 1.0
        f16 = 0.5 if e1_16[g] else 1.0
        sg = bf16_mul(rec, f8)
        sg = bf16_mul(sg, f16)
        for j in range(4 * g, 4 * g + 4):
            z = bf16_mul(xblk[j], sg)
            nibbles[j] = hif4_lane_quant(z)

    word = e6m2_byte & 0xFF
    for g in range(8):
        word |= (e1_8[g] & 1) << (8 + g)
    for g in range(16):
        word |= (e1_16[g] & 1) << (16 + g)
    return nibbles, word & 0xFFFFFFFF


def pack_fp4(nibbles) -> bytes:
    out = bytearray()
    for k in range(0, len(nibbles), 2):
        lo = nibbles[k] & 0xF
        hi = (nibbles[k + 1] & 0xF) if k + 1 < len(nibbles) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def gen_all(out_dir: Path, M: int, N: int, seed: int):
    assert N % BLOCK_SIZE == 0, "N must be multiple of 64"
    numKb = N // BLOCK_SIZE
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # input: gaussian clamp +-8, round-tripped through bf16 (what the kernel sees)
    x_bits = []
    x_val = []
    for _ in range(M * N):
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z = max(min(z, 8.0), -8.0)
        b = f32_to_bf16_bits(z)
        x_bits.append(b)
        x_val.append(bf16_bits_f32(b))

    golden_bytes = bytearray()
    scale_words = []
    for m in range(M):
        for kb in range(numKb):
            base = m * N + kb * BLOCK_SIZE
            xblk = x_val[base:base + BLOCK_SIZE]
            nibbles, word = encode_row_block(xblk)
            golden_bytes += pack_fp4(nibbles)
            scale_words.append(word)

    (out_dir / "input.bin").write_bytes(
        b"".join(struct.pack("<H", b) for b in x_bits))
    (out_dir / "golden.bin").write_bytes(bytes(golden_bytes))
    (out_dir / "scale_golden.bin").write_bytes(
        b"".join(struct.pack("<I", w) for w in scale_words))
    print(f"wrote {out_dir}/input.bin  M={M} N={N} elems={M*N} bytes={M*N*2}")
    print(f"wrote {out_dir}/golden.bin bytes={len(golden_bytes)} (M*N/2={M*N//2})")
    print(f"wrote {out_dir}/scale_golden.bin blocks={len(scale_words)} bytes={len(scale_words)*4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    args = ap.parse_args()
    gen_all(args.out_dir, args.M, args.N, args.seed)


if __name__ == "__main__":
    main()
