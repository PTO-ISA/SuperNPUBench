#!/usr/bin/env python3
"""
Data generation for conv2d_rm (RowMajor) kernel.

Key finding: The generic TLOAD (no B.DATR) on gfrun performs a 2D-aware copy
(GM(r,c) -> tile(r,c)), correctly handling BFractal ZN physical layout for
boxed tiles. No BFractal pre-conversion is needed — just use RowMajor for all
GM tensors so GetStride(3)=RowStride=Cols (correct stride).

Data layout:
  Input  : RowMajor<gM, gK> = NHWC flattened (pos * IN_C + ch)
  Weight : RowMajor<gK, gN> = B(IN_C, OUT_C) standard RowMajor = W^T
  Output : RowMajor<gM, gN> = NHWC flattened (pos * OUT_C + ch)
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os


def generate(args):
    IN_H, IN_W, IN_C, OUT_C = args.IN_H, args.IN_W, args.IN_C, args.OUT_C
    gM = IN_H * IN_W
    gN = OUT_C
    gK = IN_C

    dtype_str = args.TYPE
    np_dtype = np.float16 if dtype_str == "FP16" else np.float32
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(42)
    input_fp32 = (torch.randn((1, IN_C, IN_H, IN_W), generator=gen) * 0.1).clamp(-1, 1)
    weight_fp32 = (torch.randn((OUT_C, IN_C, 1, 1), generator=gen) * 0.1).clamp(-1, 1)

    output = F.conv2d(input_fp32, weight_fp32)

    # src0.bin: input in RowMajor<gM, gK> = NHWC flattened
    input_nhwc = input_fp32.squeeze(0).permute(1, 2, 0).reshape(gM, gK).contiguous()
    src0 = input_nhwc.numpy().astype(np_dtype)
    if dtype_str == "FP16":
        src0.view(np.int16).tofile(os.path.join(out_dir, "src0.bin"))
    else:
        src0.tofile(os.path.join(out_dir, "src0.bin"))

    # src1.bin: weight B in RowMajor<gK, gN> = (IN_C, OUT_C) RowMajor = W^T
    B = weight_fp32.reshape(OUT_C, IN_C).T.contiguous().numpy().astype(np_dtype)
    if dtype_str == "FP16":
        B.view(np.int16).tofile(os.path.join(out_dir, "src1.bin"))
    else:
        B.tofile(os.path.join(out_dir, "src1.bin"))

    # golden.bin: output in RowMajor<gM, gN> = NHWC flattened
    golden_rm = output.squeeze(0).permute(1, 2, 0).reshape(gM, gN).contiguous()
    golden = golden_rm.numpy().astype(np.float32)
    golden.tofile(os.path.join(out_dir, "golden.bin"))

    # res.bin: init with zeros
    np.zeros(gM * gN, dtype=np.float32).tofile(os.path.join(out_dir, "res.bin"))

    # all-ones data
    ones_input = np.ones(gM * gK, dtype=np_dtype)
    ones_B = np.ones((gK, gN), dtype=np_dtype)
    if dtype_str == "FP16":
        ones_input.view(np.int16).tofile(os.path.join(out_dir, "src0_ones.bin"))
        ones_B.view(np.int16).tofile(os.path.join(out_dir, "src1_ones.bin"))
    else:
        ones_input.tofile(os.path.join(out_dir, "src0_ones.bin"))
        ones_B.tofile(os.path.join(out_dir, "src1_ones.bin"))
    np.full(gM * gN, float(gK), dtype=np.float32).tofile(os.path.join(out_dir, "golden_ones.bin"))

    print(f"Generated {dtype_str} {IN_H}x{IN_W}x{IN_C}->{OUT_C} in {out_dir}")
    print(f"  src0.bin: {src0.nbytes} bytes (input NHWC RowMajor<{gM},{gK}>)")
    print(f"  src1.bin: {B.nbytes} bytes (weight standard RowMajor<{gK},{gN}>)")
    print(f"  golden.bin: {golden.nbytes} bytes (output NHWC RowMajor<{gM},{gN}>)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conv2d_rm test data (standard RowMajor)")
    parser.add_argument("--IN_H", type=int, default=4)
    parser.add_argument("--IN_W", type=int, default=4)
    parser.add_argument("--IN_C", type=int, default=16)
    parser.add_argument("--OUT_C", type=int, default=16)
    parser.add_argument("--TYPE", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--out_dir", type=str, default="./compare_dir")
    args = parser.parse_args()
    generate(args)
