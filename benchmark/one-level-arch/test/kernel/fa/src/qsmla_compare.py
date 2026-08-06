#!/usr/bin/env python3
"""
Compare NPU output (FP16, dumped from gfrun --dump-memory) with CPU golden output (FP32).
Usage: python3 qsmla_compare.py <npu_dump.bin> <golden.bin> [atol] [rtol]

NPU dump format: raw FP16 (__half, IEEE 754 half-precision, 2 bytes per element)
Golden format:   raw FP32 (float, IEEE 754 single-precision, 4 bytes per element)
"""

import sys
import struct
import math
import numpy as np

def read_fp16(filename, count=None):
    """Read binary file as float16 array, convert to float32."""
    with open(filename, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float16)
    if count is not None:
        arr = arr[:count]
    return arr.astype(np.float32)

def read_fp32(filename, count=None):
    """Read binary file as float32 array."""
    with open(filename, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float32)
    if count is not None:
        arr = arr[:count]
    return arr

def compare(npu_out, golden_out, atol=0.01, rtol=0.05):
    """Compare two float arrays with atol/rtol tolerance."""
    min_len = min(len(npu_out), len(golden_out))
    if len(npu_out) != len(golden_out):
        print(f"WARNING: length mismatch npu={len(npu_out)} golden={len(golden_out)}, comparing first {min_len}")

    max_abs_err = 0.0
    max_rel_err = 0.0
    fail_count = 0
    fail_examples = []

    for i in range(min_len):
        n_val = float(npu_out[i])
        g_val = float(golden_out[i])

        if math.isnan(n_val) or math.isnan(g_val):
            fail_count += 1
            if len(fail_examples) < 10:
                fail_examples.append((i, n_val, g_val, "NaN"))
            continue
        if math.isinf(n_val) or math.isinf(g_val):
            fail_count += 1
            if len(fail_examples) < 10:
                fail_examples.append((i, n_val, g_val, "Inf"))
            continue

        abs_err = abs(n_val - g_val)
        rel_err = abs_err / max(abs(g_val), 1e-8)

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if abs_err > atol and rel_err > rtol:
            fail_count += 1
            if len(fail_examples) < 10:
                fail_examples.append((i, n_val, g_val, abs_err, rel_err))

    print(f"=== QSMLA Precision Verification ===")
    print(f"Compared elements: {min_len}")
    print(f"Max absolute error: {max_abs_err:.8f}")
    print(f"Max relative error: {max_rel_err:.8f}")
    print(f"Tolerance: atol={atol}, rtol={rtol}")
    print(f"Failed elements: {fail_count}/{min_len}")

    if fail_examples:
        print(f"\nFirst {len(fail_examples)} failures:")
        for ex in fail_examples:
            if len(ex) == 4:
                print(f"  idx={ex[0]}: npu={ex[1]:.8f} golden={ex[2]:.8f} [{ex[3]}]")
            else:
                print(f"  idx={ex[0]}: npu={ex[1]:.8f} golden={ex[2]:.8f} abs_err={ex[3]:.8f} rel_err={ex[4]:.8f}")

    # Print first 8 values for sanity
    print(f"\nFirst 8 values comparison:")
    for i in range(min(8, min_len)):
        n = float(npu_out[i])
        g = float(golden_out[i])
        print(f"  [{i}] npu={n:.8f} golden={g:.8f} diff={abs(n-g):.8f}")

    if fail_count == 0:
        print("\n=== RESULT: PASS ===")
        return 0
    else:
        pass_rate = (min_len - fail_count) / min_len * 100
        print(f"\n=== RESULT: FAIL (pass rate: {pass_rate:.2f}%) ===")
        return 1

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 qsmla_compare.py <npu_dump_fp16.bin> <golden_fp32.bin> [atol] [rtol]")
        sys.exit(1)

    npu_file = sys.argv[1]
    golden_file = sys.argv[2]
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

    npu_out = read_fp16(npu_file)
    golden_out = read_fp32(golden_file)

    print(f"NPU output: {len(npu_out)} FP16 values from {npu_file}")
    print(f"Golden output: {len(golden_out)} FP32 values from {golden_file}")

    ret = compare(npu_out, golden_out, atol, rtol)
    sys.exit(ret)

if __name__ == '__main__':
    main()
