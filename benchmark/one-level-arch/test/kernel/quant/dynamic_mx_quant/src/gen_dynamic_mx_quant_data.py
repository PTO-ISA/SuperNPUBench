#!/usr/bin/env python3
"""Generate DynamicMxQuant host bins: input / golden output / golden scale.

Golden is a faithful Python port of the AscendC dynamic_mx_quant algorithm,
matching the specialized PTO-ISA kernels op-for-op
(kernels/quant/dynamic_mx_quant/dynamic_mx_quant_{tail,nontail}_<algo>_<dst>.hpp):

  ComputeScale (per reduction block) -> (scale_byte E8M0, recip bf16-bits)
      OCP           : max of bf16 exponent field, clamp to emax, finalize
      DYNAMIC_RANGE : add-value rounding on abs bits, emax clamp, finalize
      CUBLAS        : fp32 amax * (1/dstMax), guarded exponent extract, finalize
  ComputeData  : out = dst_dtype( x_value * bf16_value(recip_bits) )

`emax` is a property of the OUTPUT dtype (biased-domain exponent of its max
normal, <<7), mirroring AscendC GetFp8MaxExp<T>() / GetFp4MaxExp<T>():
  FP8_E4M3 -> 0x0400 (448),   FP4_E2M1 -> 0x0100 (6).

Legal (scaleAlg, dstType) pairs (AscendC dynamic_mx_quant_tiling_arch35.cpp):
  OCP -> FP8 & FP4 ;  CUBLAS -> FP8 only ;  DYNAMIC_RANGE -> FP4 only.

Reduction orientation:
  tail    : matrix [rows, cols], reduce along `cols` in BlockSize=32 chunks
  nontail : matrix [rows, cols], reduce along `rows` in BlockSize=32 chunks

Output files (per ELF compare dir):
  input.bin        : rows*cols x bf16 (2 bytes each)
  golden.bin       : FP8 -> rows*cols bytes ; FP4 -> rows*cols/2 bytes (2/byte)
  scale_golden.bin : rows*cols x uint16 (broadcast E8M0 scale byte)
"""

import argparse
import math
import random
import struct
from pathlib import Path

# --- bf16 (E8M7) domain ------------------------------------------------------
BF16_EXP_MASK = 0x7F80
BF16_ABS_MASK = 0x7FFF
BF16_EXP_BIAS = 0x7F00       # recip base (0x7f00 - shared_exp)
BF16_SHR_NUM = 7
BF16_NAN_PATTERN = 0x7F81    # recip NaN
BF16_SPECIAL_EXP = 0x0040    # recip when shared_exp == 0x7f00
BF16_ADD_VALUE_MAN1 = 0x003F # DynRange rounding add-value

# emax (biased bf16 <<7) and max-normal per output dtype.
FP8_E4M3_EMAX = 0x0400
FP8_E4M3_INV_DST_MAX = 0.002232142857  # 1/448
FP4_E2M1_EMAX = 0x0100
FP8_NAN_BYTE = 0x00FF

# --- fp32 (E8M23) domain, cuBLAS path ---------------------------------------
FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_MASK = 0x007FFFFF
FP32_SHR_NUM = 23
FP32_EXP_BIAS_CUBLAS = 0x00007F00
FP32_NAN_PACK = 0x00007F81
FP32_NUMBER_254 = 0x000000FE
FP32_NUMBER_HALF = 0x00400000
CLAMP_MIN = 1e-12

BLOCK_SIZE = 32

# emax keyed by output dtype tag.
EMAX_BY_DTYPE = {"FP8": FP8_E4M3_EMAX, "FP4": FP4_E2M1_EMAX}


def f32_to_bf16_bits(x: float) -> int:
    f32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return (f32 >> 16) & 0xFFFF


def bf16_to_f32(h: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (h & 0xFFFF) << 16))[0]


# --- fp16 (E5M10) domain: kernel input dtype option --------------------------
def f32_to_fp16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", float(x)))[0]


def fp16_bits_to_f32(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f32_to_fp8_e4m3(x: float) -> int:
    if math.isnan(x):
        return 0x7F
    if math.isinf(x):
        return 0x7F if x > 0 else 0xFF
    sign = 0
    if x < 0:
        sign, x = 1, -x
    if x == 0:
        return sign << 7
    if x >= 448.0:
        return (sign << 7) | 0x7E
    if x < 2 ** -9:
        return sign << 7
    exp = math.floor(math.log2(x))
    biased_exp = exp + 7
    if biased_exp <= 0:
        # round-half-to-even (numpy.rint) to match ttk _mx_round_mantissa rint
        mant = round(x / (2 ** -6))
        if mant >= 8:
            return (sign << 7) | (1 << 3)
        return (sign << 7) | max(0, min(7, mant))
    biased_exp = max(1, min(15, biased_exp))
    # round-half-to-even (numpy.rint) to match ttk _mx_round_mantissa rint
    mant = round((x / (2 ** exp) - 1.0) * 8)
    if mant >= 8:
        mant = 0
        biased_exp += 1
        if biased_exp > 15:
            return (sign << 7) | 0x7E
    return (sign << 7) | (biased_exp << 3) | mant


# E2M1 representable magnitudes indexed by 3-bit code (sign is bit3).
FP4_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def f32_to_fp4_e2m1(x: float) -> int:
    """Round to nearest E2M1 magnitude; returns a 4-bit nibble (sign<<3 | code)."""
    if math.isnan(x):
        return 0x7  # not a real E2M1 NaN; sentinel for golden only
    sign = 0
    if x < 0 or (x == 0 and math.copysign(1.0, x) < 0):
        sign = 1
    ax = abs(x)
    if ax >= 6.0:
        code = 7
    else:
        # round-half-to-even: on an exact tie prefer the even code. For E2M1 the
        # code LSB == mantissa LSB (code = exp<<1 | mant), so even code == even
        # mantissa, matching ttk's numpy.rint on the normalized 1-bit mantissa.
        best, code = 1e30, 0
        for c, m in enumerate(FP4_E2M1_MAG):
            d = abs(ax - m)
            if d < best or (d == best and c % 2 == 0):
                best, code = d, c
    return (sign << 3) | code


# --- finalize: shared_exp + eq_inf -> (scale_byte, recip_bits) ---------------
def finalize(shared_exp: int, eq_inf: bool) -> tuple:
    shared_exp &= 0xFFFF
    scale_byte = (shared_exp >> BF16_SHR_NUM) & 0xFFFF
    if eq_inf:
        scale_byte = FP8_NAN_BYTE

    recip = (BF16_EXP_BIAS - shared_exp) & 0xFFFF
    if eq_inf:
        recip = BF16_NAN_PATTERN
    if shared_exp == 0:
        recip = 0
    if shared_exp == BF16_EXP_BIAS:
        recip = BF16_SPECIAL_EXP
    return scale_byte, recip


# scale_recip_* take the per-block fp32 VALUES actually seen by the kernel after
# it loads the input dtype. OCP / DYNAMIC_RANGE regularize each value to bf16
# exponent bits (the kernel's uint16 bf16-exp domain: bf16 = direct bits, half =
# TCVT half->bf16 TRUNC, fp32 = exp>>16 -- all collapse to f32_to_bf16_bits of the
# value for exponent purposes). cuBLAS keeps the full-precision fp32 value amax.
def scale_recip_ocp(group_vals, emax: int) -> tuple:
    exp = [(f32_to_bf16_bits(v) & BF16_EXP_MASK) for v in group_vals]
    max_exp = max(exp)
    eq_inf = (max_exp == BF16_EXP_MASK)
    max_exp = max(max_exp, emax)
    shared = (max_exp - emax) & 0xFFFF
    return finalize(shared, eq_inf)


def scale_recip_dynamic_range(group_vals, emax: int) -> tuple:
    absb = [(f32_to_bf16_bits(v) & BF16_ABS_MASK) for v in group_vals]
    max_abs = max(absb)
    xexp = max_abs & BF16_EXP_MASK
    eq_inf = (xexp == BF16_EXP_MASK)
    invalid = (xexp < emax)
    x_add = (max_abs + BF16_ADD_VALUE_MAN1) & BF16_EXP_MASK
    if invalid:
        x_add = emax
    shared = (x_add - emax) & 0xFFFF
    return finalize(shared, eq_inf)


def scale_recip_cublas(group_vals, emax: int) -> tuple:
    # cuBLAS is FP8-only; emax unused (dstMax folded into inv_dst_max).
    max_abs = max(abs(v) for v in group_vals)

    raw = f32_bits(max_abs)
    finite = raw < FP32_EXP_MASK
    eq_zero = (raw == 0)

    m = max(max_abs, CLAMP_MIN) * FP8_E4M3_INV_DST_MAX
    s32 = f32_bits(m)
    exp32 = s32 >> FP32_SHR_NUM
    man32 = s32 & FP32_MANTISSA_MASK

    p0 = (exp32 > 0) and (exp32 < FP32_NUMBER_254) and (man32 > 0)
    p1 = (exp32 == 0) and (man32 > FP32_NUMBER_HALF)
    roundup = p0 or p1

    extract = (exp32 + 1) if roundup else exp32
    if not finite:
        extract = 0xFF
    if eq_zero:
        extract = 0
    scale_byte = extract & 0xFFFF

    sh = (extract << BF16_SHR_NUM) & 0xFFFFFFFF
    half = (FP32_EXP_BIAS_CUBLAS - sh) & 0xFFFFFFFF
    if not finite:
        half = FP32_NAN_PACK
    if eq_zero:
        half = 0
    recip = half & 0xFFFF
    return scale_byte, recip


SCALE_RECIP = {
    "OCP": scale_recip_ocp,
    "DYNAMIC_RANGE": scale_recip_dynamic_range,
    "CUBLAS": scale_recip_cublas,
}


def reduction_groups(rows: int, cols: int, kernel: str):
    groups = []
    if kernel == "tail":
        numKb = cols // BLOCK_SIZE
        for m in range(rows):
            for kb in range(numKb):
                base = m * cols + kb * BLOCK_SIZE
                groups.append([base + j for j in range(BLOCK_SIZE)])
    else:  # nontail: reduce along rows
        numKb = rows // BLOCK_SIZE
        for kb in range(numKb):
            for c in range(cols):
                groups.append([(kb * BLOCK_SIZE + r) * cols + c for r in range(BLOCK_SIZE)])
    return groups


def compute_golden(x_val, rows: int, cols: int, algo: str, kernel: str, dtype: str):
    # x_val: per-element fp32 value the kernel actually operates on (already
    # round-tripped through the input dtype). The data path multiplies this value
    # by recip; the scale path regularizes it per-algo (see scale_recip_*).
    quant = [0] * (rows * cols)   # per-element quantized code (fp8 byte or fp4 nibble)
    scale = [0] * (rows * cols)   # broadcast per-element scale byte (legacy layout)
    block_scales = []             # one scale byte per reduction block, in group order
    fn = SCALE_RECIP[algo]
    emax = EMAX_BY_DTYPE[dtype]
    enc = f32_to_fp8_e4m3 if dtype == "FP8" else f32_to_fp4_e2m1
    for idx in reduction_groups(rows, cols, kernel):
        group_vals = [x_val[i] for i in idx]
        scale_byte, recip_bits = fn(group_vals, emax)
        block_scales.append(scale_byte & 0xFF)
        factor = bf16_to_f32(recip_bits)
        for i in idx:
            quant[i] = enc(x_val[i] * factor)
            scale[i] = scale_byte
    return quant, scale, block_scales


def compact_scale_bytes(block_scales, rows: int, cols: int, kernel: str) -> bytes:
    """Pack per-block E8M0 scale bytes into the GROUND-TRUTH mxScale layout
    (ttk ttk/utilities/dtypes.py mx_quantize): pad the reduce-axis block count to
    even, then — only when the reduce axis is NOT the last axis — parity-interleave
    the block rows. Padding blocks are 2**-127 whose E8M0 byte is 0x00.

    tail (reduce axis = cols = LAST axis): ttk applies pad_to_even but NO
        interleave -> planar [rows, scaleCols], scaleCols = evenAlign(K/32).
    nontail (reduce axis = rows, NOT last): ttk applies pad_to_even THEN
        interleave(axis=reduce, n_group=2) -> [scaleRows/2, cols, 2] whose byte
        order is  for g: for c: for p in (0,1): scale[block_row = 2*g + p][c],
        i.e. even/odd block-rows of each pair zipped adjacently per column.

    NOTE (kernel gap, RECORD 问题5): the current PTO-ISA nontail kernels still
    emit COMPACT PLANAR [scaleRows, cols] (no interleave) because -D__linx does
    not expose TINTERLEAVE/TDEINTERLEAVE. This golden is the ground truth, so the
    nontail scale compare will legitimately DIVERGE from those kernels until the
    interleave intrinsic lands; that divergence is the real defect, not a golden
    bug. Tail is unaffected (ground truth has no interleave there)."""
    if kernel == "tail":
        numKb = cols // BLOCK_SIZE
        scaleCols = ((numKb + 1) // 2) * 2
        out = bytearray(rows * scaleCols)  # zero-initialized -> padding stays 0
        gi = 0
        for m in range(rows):
            for kb in range(numKb):
                out[m * scaleCols + kb] = block_scales[gi]
                gi += 1
        return bytes(out)
    # nontail: block_scales are in (kb, c) order (see reduction_groups). Build the
    # even-padded planar [scaleRows, cols] first (padding block-row stays 0x00),
    # then parity-interleave the block-row axis to match ttk ground truth.
    numKb = rows // BLOCK_SIZE
    scaleRows = ((numKb + 1) // 2) * 2
    planar = [[0] * cols for _ in range(scaleRows)]  # padding rows stay 0x00
    gi = 0
    for kb in range(numKb):
        for c in range(cols):
            planar[kb][c] = block_scales[gi]
            gi += 1
    out = bytearray(scaleRows * cols)
    oi = 0
    for g in range(scaleRows // 2):
        for c in range(cols):
            for p in range(2):
                out[oi] = planar[2 * g + p][c]
                oi += 1
    return bytes(out)


def pack_output(quant, dtype: str) -> bytes:
    if dtype == "FP8":
        return bytes(quant)
    # FP4: pack 2 nibbles/byte, low nibble first (element 2k in low, 2k+1 in high).
    out = bytearray()
    for k in range(0, len(quant), 2):
        lo = quant[k] & 0xF
        hi = (quant[k + 1] & 0xF) if k + 1 < len(quant) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


# Per input-dtype: round-trip the true fp32 value through the input dtype (what
# the kernel sees after TLOAD) and pack input.bin in that dtype's byte layout.
def input_encode(x_f32, in_dtype: str):
    if in_dtype == "bf16":
        bits = [f32_to_bf16_bits(v) for v in x_f32]
        x_val = [bf16_to_f32(b) for b in bits]
        raw = b"".join(struct.pack("<H", b) for b in bits)
    elif in_dtype == "fp16":
        bits = [f32_to_fp16_bits(v) for v in x_f32]
        x_val = [fp16_bits_to_f32(b) for b in bits]
        raw = b"".join(struct.pack("<H", b) for b in bits)
    else:  # fp32
        x_val = [struct.unpack("<f", struct.pack("<f", float(v)))[0] for v in x_f32]
        raw = b"".join(struct.pack("<f", v) for v in x_val)
    return x_val, raw


def gen_all(out_dir: Path, rows: int, cols: int, algo: str, kernel: str,
            dtype: str, seed: int, scale_layout: str, in_dtype: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    x_f32 = []
    for _ in range(rows * cols):
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        x_f32.append(max(min(z, 8.0), -8.0))
    x_val, input_raw = input_encode(x_f32, in_dtype)
    quant, scale, block_scales = compute_golden(x_val, rows, cols, algo, kernel, dtype)
    golden = pack_output(quant, dtype)

    (out_dir / "input.bin").write_bytes(input_raw)
    (out_dir / "golden.bin").write_bytes(golden)
    print(f"wrote {out_dir}/input.bin  in_dtype={in_dtype} elems={rows*cols} bytes={len(input_raw)}")
    print(f"wrote {out_dir}/golden.bin dtype={dtype} bytes={len(golden)}")
    if scale_layout == "compact":
        sbytes = compact_scale_bytes(block_scales, rows, cols, kernel)
        (out_dir / "scale_golden.bin").write_bytes(sbytes)
        print(f"wrote {out_dir}/scale_golden.bin layout=compact uint8 bytes={len(sbytes)}")
    else:
        (out_dir / "scale_golden.bin").write_bytes(
            b"".join(struct.pack("<H", v) for v in scale))
        print(f"wrote {out_dir}/scale_golden.bin layout=broadcast uint16 "
              f"elems={len(scale)} bytes={len(scale)*2}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8, help="rows")
    parser.add_argument("--K", type=int, default=32, help="cols")
    parser.add_argument("--algo", type=str, default="OCP",
                        choices=["OCP", "CUBLAS", "DYNAMIC_RANGE"])
    parser.add_argument("--kernel", type=str, default="tail",
                        choices=["tail", "nontail"])
    parser.add_argument("--dtype", type=str, default="FP8",
                        choices=["FP8", "FP4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-layout", type=str, default="broadcast",
                        choices=["broadcast", "compact"],
                        help="broadcast=legacy uint16 per-elem; compact=AscendC uint8 [M,scaleCols]")
    parser.add_argument("--in-dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="input.bin dtype: bf16 (2B) | fp16/E5M10 (2B) | fp32 (4B)")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()
    gen_all(args.out_dir, args.M, args.K, args.algo, args.kernel, args.dtype,
            args.seed, args.scale_layout, args.in_dtype)


if __name__ == "__main__":
    main()
