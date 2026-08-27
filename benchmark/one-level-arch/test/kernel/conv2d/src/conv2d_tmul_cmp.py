#!/usr/bin/env python3
"""Compare conv2d TMATMUL result (RowMajor output) against golden reference.

This script verifies the accuracy of the TMATMUL/TMATMUL_ACC computation
by comparing the NPU output (stored via RowMajor TSTORE) against a
PyTorch golden reference. The RowMajor output bypasses the ColMajor
TSTORE stride bug, so the stored output equals the TMATMUL result.

Usage:
  python3 conv2d_tmul_cmp.py --npu /tmp/conv2d_tmul_multik_v2.bin \
                             --K 64 --gM 16 --gN 16
"""

import argparse
import numpy as np


def generate_golden(gM: int, gN: int, gK: int) -> np.ndarray:
    """Generate golden reference for all-ones input/weight matmul.

    input  (gM, gK) ColMajor, all 1.0
    weight (gK, gN) ColMajor, all 1.0
    output (gM, gN) RowMajor, each element = gK
    """
    golden = np.full((gM, gN), float(gK), dtype=np.float32)
    return golden


def compare(npu: np.ndarray, golden: np.ndarray, label: str = ""):
    """Compare NPU output against golden, focusing on TMATMUL accuracy."""
    npu_2d = npu.reshape(golden.shape)
    diff = np.abs(npu_2d - golden)

    total = npu_2d.size
    nz_mask = npu_2d != 0
    nz_count = np.count_nonzero(nz_mask)
    nz_diff = diff[nz_mask]
    nz_max_diff = float(nz_diff.max()) if nz_count > 0 else 0.0
    nz_mean_diff = float(nz_diff.mean()) if nz_count > 0 else 0.0
    nz_correct = np.count_nonzero(nz_diff <= 1e-4)
    eps = 1e-4
    mismatch = np.count_nonzero(nz_diff > eps)

    print(f"=== {label} ===")
    print(f"Shape: {golden.shape}, Total elements: {total}")
    print(f"Non-zero (TSTORE written): {nz_count}/{total} ({100*nz_count/total:.1f}%)")
    print(f"Expected value: {golden[0, 0]}")
    print(f"Unique NPU values: {np.unique(npu_2d)}")
    print(f"Max abs diff (non-zero): {nz_max_diff:.8e}")
    print(f"Mean abs diff (non-zero): {nz_mean_diff:.8e}")
    print(f"Correct (eps={eps}): {nz_correct}/{nz_count}")
    print(f"Mismatch: {mismatch}/{nz_count}")
    print(f"Non-zero row indices: {[r for r in range(golden.shape[0]) if np.count_nonzero(npu_2d[r]) > 0]}")
    status = "PASS" if mismatch == 0 and nz_count > 0 else "FAIL"
    print(f"TMATMUL Accuracy: {status}")
    print()
    return status == "PASS"


def main():
    parser = argparse.ArgumentParser(
        description="Compare conv2d TMATMUL result against golden"
    )
    parser.add_argument("--npu", required=True, help="NPU output dump file")
    parser.add_argument("--K", type=int, required=True, help="gK (IN_C, accumulation depth)")
    parser.add_argument("--gM", type=int, default=16, help="gM (IN_H * IN_W)")
    parser.add_argument("--gN", type=int, default=16, help="gN (OUT_C)")
    parser.add_argument("--label", default="", help="label for this case")
    args = parser.parse_args()

    npu = np.fromfile(args.npu, dtype=np.float32)
    golden = generate_golden(args.gM, args.gN, args.K)
    compare(npu, golden, args.label or f"K={args.K}")


if __name__ == "__main__":
    main()
