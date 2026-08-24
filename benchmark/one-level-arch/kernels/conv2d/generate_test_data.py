#!/usr/bin/env python3
"""
Generate test data for conv2d verification (NCHW format).

This script creates input feature map (NCHW) and weights, performs convolution using PyTorch,
and saves the data in binary format for SuperNPUBench conv2d kernel testing.
"""

import torch
import numpy as np
import argparse
import os

def compute_output_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    """Compute output size for a single dimension"""
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def main():
    parser = argparse.ArgumentParser(description='Generate conv2d test data (NCHW format)')
    parser.add_argument('--in-h', type=int, default=14, help='Input height')
    parser.add_argument('--in-w', type=int, default=14, help='Input width')
    parser.add_argument('--in-c', type=int, default=64, help='Input channels')
    parser.add_argument('--out-c', type=int, default=64, help='Output channels')
    parser.add_argument('--kernel-h', type=int, default=3, help='Kernel height')
    parser.add_argument('--kernel-w', type=int, default=3, help='Kernel width')
    parser.add_argument('--stride-h', type=int, default=1, help='Stride height')
    parser.add_argument('--stride-w', type=int, default=1, help='Stride width')
    parser.add_argument('--pad-h', type=int, default=0, help='Padding height')
    parser.add_argument('--pad-w', type=int, default=0, help='Padding width')
    parser.add_argument('--dilation-h', type=int, default=1, help='Dilation height')
    parser.add_argument('--dilation-w', type=int, default=1, help='Dilation width')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16'], help='Data type')
    parser.add_argument('--output-dir', type=str, default='./test_data', help='Output directory')
    
    args = parser.parse_args()
    
    # Compute output dimensions
    out_h = compute_output_size(args.in_h, args.kernel_h, args.stride_h, args.pad_h, args.dilation_h)
    out_w = compute_output_size(args.in_w, args.kernel_w, args.stride_w, args.pad_w, args.dilation_w)
    
    print(f"Input shape: {args.in_c}×{args.in_h}×{args.in_w} (NCHW)")
    print(f"Kernel: {args.kernel_h}×{args.kernel_w}")
    print(f"Stride: {args.stride_h}×{args.stride_w}")
    print(f"Padding: {args.pad_h}×{args.pad_w}")
    print(f"Dilation: {args.dilation_h}×{args.dilation_w}")
    print(f"Output shape: {args.out_c}×{out_h}×{out_w} (NCHW)")
    
    # Create input and weight tensors
    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    input_nchw = torch.randn(1, args.in_c, args.in_h, args.in_w, dtype=dtype)
    weight = torch.randn(args.out_c, args.in_c, args.kernel_h, args.kernel_w, dtype=dtype)
    
    # Perform convolution using PyTorch
    conv_layer = torch.nn.Conv2d(
        args.in_c, args.out_c, 
        kernel_size=(args.kernel_h, args.kernel_w),
        stride=(args.stride_h, args.stride_w),
        padding=(args.pad_h, args.pad_w),
        dilation=(args.dilation_h, args.dilation_w),
        bias=False
    )
    conv_layer.weight.data = weight
    output_ref = conv_layer(input_nchw)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save binary files
    # Input: NCHW format (C, H, W) for batch=1
    if args.dtype == 'fp32':
        input_nchw.squeeze(0).numpy().astype(np.float32).tofile(
            os.path.join(args.output_dir, 'input_nchw.bin'))
        weight.numpy().astype(np.float32).tofile(
            os.path.join(args.output_dir, 'weight.bin'))
        output_ref.squeeze(0).numpy().astype(np.float32).tofile(
            os.path.join(args.output_dir, 'output_ref.bin'))
    else:
        input_nchw.squeeze(0).numpy().astype(np.float16).tofile(
            os.path.join(args.output_dir, 'input_nchw.bin'))
        weight.numpy().astype(np.float16).tofile(
            os.path.join(args.output_dir, 'weight.bin'))
        output_ref.squeeze(0).numpy().astype(np.float32).tofile(
            os.path.join(args.output_dir, 'output_ref.bin'))
    
    print(f"\nTest data saved to {args.output_dir}/")
    print("  - input_nchw.bin: Input feature map (NCHW format)")
    print("  - weight.bin: Convolution weights")
    print("  - output_ref.bin: Reference output from PyTorch")
    
    # Print compile command
    print(f"\nCompile command:")
    print(f"  make TYPE={args.dtype.upper().replace('FP', 'CONV_')} \\")
    print(f"       IN_H={args.in_h} IN_W={args.in_w} IN_C={args.in_c} OUT_C={args.out_c} \\")
    print(f"       KH={args.kernel_h} KW={args.kernel_w} \\")
    print(f"       STRIDE_H={args.stride_h} STRIDE_W={args.stride_w} \\")
    print(f"       PAD_H={args.pad_h} PAD_W={args.pad_w} \\")
    print(f"       DILATION_H={args.dilation_h} DILATION_W={args.dilation_w} \\")
    print(f"       tilM=16 tilN=16 tilK=16 \\")
    print(f"       TESTCASE=conv2d")

if __name__ == '__main__':
    main()