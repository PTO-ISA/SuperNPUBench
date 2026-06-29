#!/usr/bin/env python3
"""
Generate topk dataset for testing.
- Input: 131072 uint16_t values (random, seed=42)
- Output: top 2048 values (sorted descending = largest values)
"""

import struct
import random

def main():
    NUM_INPUT = 131072
    TOP_K = 2048
    OUTPUT_DIR = "data_obj"

    print(f"Generating topk dataset:")
    print(f"  Input elements:  {NUM_INPUT} uint16_t")
    print(f"  Top-K:           {TOP_K}")

    # Generate random uint16_t values
    random.seed(42)
    values = [random.randint(0, 65535) for _ in range(NUM_INPUT)]

    # Compute expected top 2048 (largest values, unsigned comparison)
    sorted_values = sorted(values, reverse=True)
    expected_topk = sorted_values[:TOP_K]

    # Write input data
    with open(f"{OUTPUT_DIR}/input_131072.data", "wb") as f:
        for v in values:
            f.write(struct.pack("<H", v))

    # Write expected output (top 2048 sorted descending)
    with open(f"{OUTPUT_DIR}/top_2048_out.data", "wb") as f:
        for v in expected_topk:
            f.write(struct.pack("<H", v))

    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    print(f"  input_131072.data:  {NUM_INPUT * 2} bytes ({NUM_INPUT} values)")
    print(f"  top_2048_out.data:  {TOP_K * 2} bytes ({TOP_K} values)")
    print(f"\nSample input (first 10): {[hex(v) for v in values[:10]]}")
    print(f"Sample output (first 10): {[hex(v) for v in expected_topk[:10]]}")

    # Verify
    all_sorted = sorted(values, reverse=True)
    assert expected_topk == all_sorted[:TOP_K], "TopK mismatch!"
    print("\nVerification: PASS")


if __name__ == "__main__":
    main()