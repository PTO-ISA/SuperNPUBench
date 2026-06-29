#!/usr/bin/env python3
"""
Generate hashtable_lookup dataset.
- Table capacity: 2,550,000 entries (non-power-of-2, uses modulo)
- Load factor: 83% (2,116,500 keys inserted)
- Query keys: 6144 (random sample from inserted keys, 100% found)
- Per-warp(64) target: mean~40, median~32, max~300
"""

import struct
import random

# MurmurHash3 constants (must match the implementation in hashtable_lookup_simd.cpp)
C1 = 0xcc9e2d51
C2 = 0x1b873593
C3 = 0xe6546b64

def murmurhash3(key, seed=0):
    """MurmurHash3 for 64-bit key (unsigned), returns 32-bit hash.
    Must match the C++ implementation which processes both low and high 32 bits."""
    key_low = key & 0xFFFFFFFF
    key_high = (key >> 32) & 0xFFFFFFFF

    # Block 1: Process low 32 bits
    k1 = key_low
    k1 = (k1 * C1) & 0xFFFFFFFF
    k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # rotl 15
    k1 = (k1 * C2) & 0xFFFFFFFF

    h = seed ^ k1
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF  # rotl 13
    h = (h * 5 + C3) & 0xFFFFFFFF

    # Block 2: Process high 32 bits
    k1 = key_high
    k1 = (k1 * C1) & 0xFFFFFFFF
    k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # rotl 15
    k1 = (k1 * C2) & 0xFFFFFFFF

    h ^= k1
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF  # rotl 13
    h = (h * 5 + C3) & 0xFFFFFFFF

    # Finalization
    h ^= 8  # len = 8 bytes
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16

    return h


def main():
    CAP = 2550000
    LOAD_FACTOR = 0.83
    NUM_INSERTED = int(CAP * LOAD_FACTOR)  # 2116500
    NUM_QUERIES = 6144
    SEED_TABLE = 42   # Seed for key generation / table building
    SEED_QUERY = 211  # Seed for query key sampling (tuned for target per-warp stats)

    print(f"Generating hashtable_lookup dataset:")
    print(f"  Table capacity:  {CAP}")
    print(f"  Keys inserted:   {NUM_INSERTED} ({LOAD_FACTOR*100:.0f}% load factor)")
    print(f"  Query keys:      {NUM_QUERIES}")
    print(f"  Entry size:      16 bytes (key:int64 + value:int32 + padding)")
    print(f"  Table size:      {CAP * 16} bytes ({CAP * 16 / 1024 / 1024:.1f} MB)")

    # Generate random 64-bit keys
    random.seed(SEED_TABLE)
    print("\nGenerating random keys...")
    inserted_keys = []
    seen = set()
    while len(inserted_keys) < NUM_INSERTED:
        key = random.getrandbits(64)
        if key not in seen:
            seen.add(key)
            inserted_keys.append(key)
    print(f"  Generated {len(inserted_keys)} unique keys")

    # Build hashtable using linear probing
    EMPTY_KEY = 0x8000000000000000  # Marker for empty slot
    table = [(EMPTY_KEY, -1, 0) for _ in range(CAP)]  # (key, value, padding)

    key_to_value = {}
    print("Building hash table (linear probing)...")
    for i, key in enumerate(inserted_keys):
        if i % 500000 == 0:
            print(f"  Inserted {i}/{NUM_INSERTED}...")
        value = i + 1  # Value is 1-based index
        idx = murmurhash3(key) % CAP  # Modulo for non-power-of-2

        # Linear probe
        while True:
            if table[idx][0] == EMPTY_KEY:
                table[idx] = (key, value, 0)
                key_to_value[key] = value
                break
            idx = (idx + 1) % CAP
    print(f"  Hash table built. {len(key_to_value)} keys inserted.")

    # Compute probe distances for all inserted keys
    print("\nComputing probe distances for all inserted keys...")
    key_probes = {}
    for i, key in enumerate(inserted_keys):
        if i % 500000 == 0:
            print(f"  Probing {i}/{NUM_INSERTED}...")
        h = murmurhash3(key)
        idx = h % CAP
        for probes in range(1, CAP + 1):
            if table[idx][0] == key:
                key_probes[key] = probes - 1  # 0-based: 0 means found at hash slot
                break
            idx = (idx + 1) % CAP

    all_probes = list(key_probes.values())
    print(f"\n=== Per-key probe statistics ({len(all_probes)} keys) ===")
    print(f"  Min: {min(all_probes)}")
    print(f"  Max: {max(all_probes)}")
    print(f"  Mean: {sum(all_probes)/len(all_probes):.2f}")

    # Sample query keys from all inserted keys
    # SEED_QUERY is tuned so per-warp(64) stats hit target: mean~40, median~32, max~300
    print(f"\nSampling {NUM_QUERIES} query keys from inserted keys (seed={SEED_QUERY})...")
    random.seed(SEED_QUERY)
    query_keys = random.sample(inserted_keys, NUM_QUERIES)
    expected_values = [key_to_value[key] for key in query_keys]

    def u64_to_i64(u):
        """Convert unsigned 64-bit to signed int64 for struct packing."""
        if u >= (1 << 63):
            return u - (1 << 64)
        return u

    # Write binary data files
    output_dir = "data_obj"
    print(f"\nWriting data files to {output_dir}/...")
    with open(f"{output_dir}/inserted_slot.data", "wb") as f:
        for key, value, padding in table:
            f.write(struct.pack("<q", u64_to_i64(key)))
            f.write(struct.pack("<i", value))
            f.write(struct.pack("<i", padding))

    with open(f"{output_dir}/lookup_keys.data", "wb") as f:
        for key in query_keys:
            f.write(struct.pack("<q", u64_to_i64(key)))

    with open(f"{output_dir}/lookup_values.data", "wb") as f:
        for val in expected_values:
            f.write(struct.pack("<i", val))

    print(f"\nGenerated files in {output_dir}/:")
    print(f"  inserted_slot.data:  {CAP * 16} bytes ({CAP} entries, {CAP * 16 / 1024 / 1024:.1f} MB)")
    print(f"  lookup_keys.data:    {NUM_QUERIES * 8} bytes ({NUM_QUERIES} keys)")
    print(f"  lookup_values.data:  {NUM_QUERIES * 4} bytes ({NUM_QUERIES} values)")

    # Per-warp probe statistics
    query_probes = [key_probes[k] for k in query_keys]
    for warp_size in [32, 64]:
        num_warps = (NUM_QUERIES + warp_size - 1) // warp_size
        warp_maxes = []
        warp_means = []
        for w in range(num_warps):
            start = w * warp_size
            end = min(start + warp_size, NUM_QUERIES)
            wp = query_probes[start:end]
            warp_maxes.append(max(wp))
            warp_means.append(sum(wp) / len(wp))

        print(f"\n=== Per-warp stats (warp={warp_size}, {num_warps} warps) ===")
        print(f"  Warp max probe:  min={min(warp_maxes)}, max={max(warp_maxes)}, mean={sum(warp_maxes)/len(warp_maxes):.2f}")
        print(f"  Warp mean probe: min={min(warp_means):.2f}, max={max(warp_means):.2f}, mean={sum(warp_means)/len(warp_means):.2f}")
        print(f"\n  {'Warp':>4}  {'MaxProbe':>8}  {'MeanProbe':>9}  {'MinInWarp':>9}")
        for i, (mx, mu) in enumerate(zip(warp_maxes, warp_means)):
            start = i * warp_size; end = min(start + warp_size, NUM_QUERIES)
            print(f"  {i:4d}  {mx:8d}  {mu:9.2f}  {min(query_probes[start:end]):9d}")


if __name__ == "__main__":
    main()
