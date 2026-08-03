#!/usr/bin/env python3
"""Generate DynamicMxQuant host bins: input / golden output / golden scale.

Generates bf16 input data and computes golden fp8 e4m3 output + scale_byte
for 3 scale algorithms: OCP, CUBLAS, DYNAMIC_RANGE.

Output files (per ELF):
  input.bin          : M*K x bf16 (2 bytes each)
  golden.bin         : M*K x fp8_e4m3 (1 byte each)
  scale_golden.bin   : M x numKb x BlockSize x uint16 (2 bytes each, padded)
"""

import argparse
import math
import os
import random
import struct
from pathlib import Path

BF16_EXP_MASK = 0x7F80
BF16_ABS_MASK = 0x7FFF
FP8_E4M3_EMAX = 0x0400
FP8_E4M3_DST_MAX = 448.0
FP8_E4M3_INV_DST_MAX = 1.0 / 448.0
FP8_NAN_BYTE = 0x00FF
FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_MASK = 0x007FFFFF
FP32_SHR_NUM = 23
BF16_SHR_NUM = 7
CLAMP_MIN = 1e-12
BLOCK_SIZE = 32
KSCALE_STRIDE = 8
ADD_VALUE = 0x003F


def f32_to_bf16_bits(x: float) -> int:
    b = struct.pack("<f", float(x))
    f32 = struct.unpack("<I", b)[0]
    return (f32 >> 16) & 0xFFFF


def bf16_bits_to_f32(h: int) -> float:
    f32_bits = h << 16
    b = struct.pack("<I", f32_bits)
    return struct.unpack("<f", b)[0]


def f32_to_fp8_e4m3(x: float) -> int:
    if math.isnan(x):
        return 0x7F
    if math.isinf(x):
        return 0x7F if x > 0 else 0xFF
    sign = 0
    if x < 0:
        sign = 1
        x = -x
    if x == 0:
        return sign << 7
    if x >= 448.0:
        return (sign << 7) | 0x7E
    if x < 2**-9:
        return sign << 7
    exp = math.floor(math.log2(x))
    bias = 7
    biased_exp = exp + bias
    if biased_exp <= 0:
        mant = int(x / (2**(-6)) + 0.5)
        if mant >= 8:
            biased_exp = 1
            mant = 0
        else:
            mant = max(0, min(7, mant))
        return (sign << 7) | (mant << 0)
    biased_exp = max(1, min(15, biased_exp))
    mant_f = x / (2**exp) - 1.0
    mant = int(mant_f * 8 + 0.5)
    if mant >= 8:
        mant = 0
        biased_exp += 1
        if biased_exp > 15:
            return (sign << 7) | 0x7E
    return (sign << 7) | (biased_exp << 3) | mant


def compute_ocp_scale_byte(x_bf16_block: list[int]) -> list[int]:
    num_rows = len(x_bf16_block) // BLOCK_SIZE
    scale_bytes = []
    for r in range(num_rows):
        row = x_bf16_block[r * BLOCK_SIZE : (r + 1) * BLOCK_SIZE]
        exp_bits = [(v & BF16_EXP_MASK) for v in row]
        max_exp = max(exp_bits)
        if max_exp == BF16_EXP_MASK:
            for _ in range(BLOCK_SIZE):
                scale_bytes.append(FP8_NAN_BYTE)
            continue
        max_exp = max(max_exp, FP8_E4M3_EMAX)
        shared_exp = max_exp - FP8_E4M3_EMAX
        sb = shared_exp >> BF16_SHR_NUM
        for _ in range(BLOCK_SIZE):
            scale_bytes.append(sb)
    return scale_bytes


def compute_cublas_scale_byte(x_bf16_block: list[int]) -> list[int]:
    num_rows = len(x_bf16_block) // BLOCK_SIZE
    scale_bytes = []
    for r in range(num_rows):
        row = x_bf16_block[r * BLOCK_SIZE : (r + 1) * BLOCK_SIZE]
        x_f32 = [bf16_bits_to_f32(v) for v in row]
        abs_x = [abs(v) for v in x_f32]
        max_abs = max(abs_x)
        s = max_abs * FP8_E4M3_INV_DST_MAX
        s_bits = struct.unpack("<I", struct.pack("<f", s))[0]
        exp_bits = (s_bits >> FP32_SHR_NUM) & 0xFF
        man_bits = s_bits & FP32_MANTISSA_MASK
        if man_bits != 0:
            exp_bits += 1
        sb = exp_bits & 0xFF
        for _ in range(BLOCK_SIZE):
            scale_bytes.append(sb)
    return scale_bytes


def compute_dynamic_range_scale_byte(x_bf16_block: list[int]) -> list[int]:
    num_rows = len(x_bf16_block) // BLOCK_SIZE
    scale_bytes = []
    for r in range(num_rows):
        row = x_bf16_block[r * BLOCK_SIZE : (r + 1) * BLOCK_SIZE]
        abs_x = [(v & BF16_ABS_MASK) for v in row]
        max_abs = max(abs_x)
        max_abs_rounded = (max_abs + ADD_VALUE) & 0xFFFF
        exp_bits = max_abs_rounded & BF16_EXP_MASK
        if exp_bits == BF16_EXP_MASK:
            for _ in range(BLOCK_SIZE):
                scale_bytes.append(FP8_NAN_BYTE)
            continue
        exp_bits = max(exp_bits, FP8_E4M3_EMAX)
        shared_exp = exp_bits - FP8_E4M3_EMAX
        sb = shared_exp >> BF16_SHR_NUM
        for _ in range(BLOCK_SIZE):
            scale_bytes.append(sb)
    return scale_bytes


def compute_golden(x_bf16: list[int], M: int, K: int, algo: str) -> tuple[list[int], list[int]]:
    numKb = K // BLOCK_SIZE
    output = []
    scale_padded = []
    for m in range(M):
        for kb in range(numKb):
            block_start = m * K + kb * BLOCK_SIZE
            block = x_bf16[block_start : block_start + BLOCK_SIZE]
            if algo == "OCP":
                sb = compute_ocp_scale_byte(block)
            elif algo == "CUBLAS":
                sb = compute_cublas_scale_byte(block)
            else:
                sb = compute_dynamic_range_scale_byte(block)
            x_f32 = [bf16_bits_to_f32(v) for v in block]
            abs_x = [abs(v) for v in x_f32]
            amax = max(abs_x)
            amax = max(amax, CLAMP_MIN)
            sfinv = FP8_E4M3_DST_MAX / amax
            for v in x_f32:
                outf = v * sfinv
                output.append(f32_to_fp8_e4m3(outf))
            for _ in range(BLOCK_SIZE):
                scale_padded.append(sb[0])
    return output, scale_padded


def gen_all(
    out_dir: Path,
    M: int,
    K: int,
    algo: str,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    x_f32 = []
    for _ in range(M * K):
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        x_f32.append(max(min(z, 8.0), -8.0))
    x_bf16 = [f32_to_bf16_bits(v) for v in x_f32]
    output, scale_padded = compute_golden(x_bf16, M, K, algo)
    (out_dir / "input.bin").write_bytes(
        b"".join(struct.pack("<H", v) for v in x_bf16)
    )
    (out_dir / "golden.bin").write_bytes(
        bytes(output)
    )
    (out_dir / "scale_golden.bin").write_bytes(
        b"".join(struct.pack("<H", v) for v in scale_padded)
    )
    print(f"wrote {out_dir}/input.bin  elems={M*K} bytes={M*K*2}")
    print(f"wrote {out_dir}/golden.bin elems={M*K} bytes={M*K}")
    print(f"wrote {out_dir}/scale_golden.bin elems={len(scale_padded)} bytes={len(scale_padded)*2}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--algo", type=str, default="OCP",
                        choices=["OCP", "CUBLAS", "DYNAMIC_RANGE"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()
    gen_all(args.out_dir, args.M, args.K, args.algo, args.seed)


if __name__ == "__main__":
    main()
