#!/usr/bin/env python3
"""
Data generation for conv2d_rm_mt (RowMajor, multi-thread batch-of-4) kernel.

Batch-level parallelism: 4 PE inputs convolved with one shared weight
(the matmul_multithread pattern applied to conv2d 1x1).

Data layout:
  src0.bin   : 4 PE inputs concatenated, each NHWC flattened
               RowMajor<gM, gK> (pos * IN_C + ch), total [4*gM, gK]
  src1.bin   : shared weight B = W^T in RowMajor<gK, gN> (IN_C, OUT_C)
  golden.bin : 4 PE outputs concatenated, each NHWC flattened
               RowMajor<gM, gN> (pos * OUT_C + ch), total [4*gM, gN]

golden reference: torch.nn.functional.conv2d (1x1, no padding) on the
batch of 4 inputs.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os


def generate(args):
    IN_H, IN_W, IN_C, OUT_C = args.IN_H, args.IN_W, args.IN_C, args.OUT_C
    kPeNum = 4
    gM = IN_H * IN_W
    gN = OUT_C
    gK = IN_C

    dtype_str = args.TYPE
    np_dtype = np.float16 if dtype_str == "FP16" else np.float32
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(42)
    # Batch of 4 inputs (one per PE), shared weight
    input_fp32 = (torch.randn((kPeNum, IN_C, IN_H, IN_W), generator=gen) * 0.1).clamp(-1, 1)
    weight_fp32 = (torch.randn((OUT_C, IN_C, 1, 1), generator=gen) * 0.1).clamp(-1, 1)

    output = F.conv2d(input_fp32, weight_fp32)  # (4, OUT_C, IN_H, IN_W)

    # src0.bin: 4 PE inputs, each NHWC flattened, concatenated -> [4*gM, gK]
    input_nhwc = input_fp32.permute(0, 2, 3, 1).reshape(kPeNum * gM, gK).contiguous()
    src0 = input_nhwc.numpy().astype(np_dtype)
    if dtype_str == "FP16":
        src0.view(np.int16).tofile(os.path.join(out_dir, "src0.bin"))
    else:
        src0.tofile(os.path.join(out_dir, "src0.bin"))

    # src1.bin: shared weight B in RowMajor<gK, gN> = (IN_C, OUT_C) RowMajor = W^T
    B = weight_fp32.reshape(OUT_C, IN_C).T.contiguous().numpy().astype(np_dtype)
    if dtype_str == "FP16":
        B.view(np.int16).tofile(os.path.join(out_dir, "src1.bin"))
    else:
        B.tofile(os.path.join(out_dir, "src1.bin"))

    # golden.bin: 4 PE outputs, each NHWC flattened, concatenated -> [4*gM, gN]
    golden_rm = output.permute(0, 2, 3, 1).reshape(kPeNum * gM, gN).contiguous()
    golden = golden_rm.numpy().astype(np.float32)
    golden.tofile(os.path.join(out_dir, "golden.bin"))

    # res.bin: init with zeros
    np.zeros(kPeNum * gM * gN, dtype=np.float32).tofile(os.path.join(out_dir, "res.bin"))

    # all-ones data (matches the volatile-init verification values)
    ones_input = np.ones(kPeNum * gM * gK, dtype=np_dtype)
    ones_B = np.ones((gK, gN), dtype=np_dtype)
    if dtype_str == "FP16":
        ones_input.view(np.int16).tofile(os.path.join(out_dir, "src0_ones.bin"))
        ones_B.view(np.int16).tofile(os.path.join(out_dir, "src1_ones.bin"))
    else:
        ones_input.tofile(os.path.join(out_dir, "src0_ones.bin"))
        ones_B.tofile(os.path.join(out_dir, "src1_ones.bin"))
    np.full(kPeNum * gM * gN, float(gK), dtype=np.float32).tofile(os.path.join(out_dir, "golden_ones.bin"))

    print(f"Generated {dtype_str} batch={kPeNum} {IN_H}x{IN_W}x{IN_C}->{OUT_C} in {out_dir}")
    print(f"  src0.bin: {src0.nbytes} bytes (4 PE inputs NHWC RowMajor<{gM},{gK}> each)")
    print(f"  src1.bin: {B.nbytes} bytes (shared weight standard RowMajor<{gK},{gN}>)")
    print(f"  golden.bin: {golden.nbytes} bytes (4 PE outputs NHWC RowMajor<{gM},{gN}> each)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conv2d_rm_mt test data (batch-of-4, standard RowMajor)")
    parser.add_argument("--IN_H", type=int, default=4)
    parser.add_argument("--IN_W", type=int, default=4)
    parser.add_argument("--IN_C", type=int, default=16)
    parser.add_argument("--OUT_C", type=int, default=16)
    parser.add_argument("--TYPE", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--out_dir", type=str, default="./compare_dir")
    args = parser.parse_args()
    generate(args)
