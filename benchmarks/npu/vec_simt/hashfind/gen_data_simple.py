#!/usr/bin/env python3
"""
Generate simple hashfind dataset for testing uint16_t offset limit.
- Table capacity: 8192 entries
- Load factor: 80% (6554 keys inserted)
- Query keys: 1024 (80% found, 20% not found)
"""

import struct
import random

# MurmurHash3 constants (must match the implementation in hashfind.cpp)
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
    CAP = 2048
    NUM_INSERTED = int(CAP * 0.8)  # 1638
    NUM_QUERIES = 1024
    FOUND_RATIO = 0.8

    print(f"Generating hashfind simple dataset:")
    print(f"  Table capacity: {CAP}")
    print(f"  Keys inserted: {NUM_INSERTED}")
    print(f"  Query keys: {NUM_QUERIES}")
    print(f"  Entry size: 16 bytes (key:int64 + value:int32 + padding)")
    print(f"  Max offset: {CAP * 16} bytes (exceeds uint16_t={65535})")

    # Generate random 64-bit keys as unsigned, then convert for storage
    random.seed(42)  # Deterministic for reproducibility
    inserted_keys = []
    seen = set()
    while len(inserted_keys) < NUM_INSERTED:
        key = random.getrandbits(64)
        if key not in seen:
            seen.add(key)
            # Store as unsigned, will convert when packing
            inserted_keys.append(key)

    inserted_keys = list(inserted_keys)

    # Build hashtable using linear probing
    empty_key = 0x8000000000000000  # Marker for empty slot
    empty_value = -1  # Use -1 for empty slot value (will be stored as 0xFFFFFFFF)
    table = [(empty_key, empty_value, 0) for _ in range(CAP)]  # (key, value, padding)

    key_to_value = {}
    for i, key in enumerate(inserted_keys):
        value = i + 1  # Value is 1-based index
        idx = murmurhash3(key) & (CAP - 1)  # Mask for power-of-2

        # Linear probe
        while True:
            if table[idx][0] == 0x8000000000000000:  # Empty slot
                table[idx] = (key, value, 0)
                key_to_value[key] = value
                break
            idx = (idx + 1) & (CAP - 1)

    # Generate query keys - all from inserted keys (100% found)
    # Filter to only include keys with probing distance < 10
    # This ensures first 256 keys all have probe distance < 10

    query_keys = []
    expected_values = []

    # Build a map of probing distances for all inserted keys
    print("\nCalculating probing distances...")
    key_probes = {}  # key -> probe distance

    for key in inserted_keys:
        h = murmurhash3(key)
        idx = h & (CAP - 1)
        probes = 0
        found_idx = -1

        for probes in range(1, CAP + 1):
            entry = table[idx]
            stored_key = entry[0]
            if stored_key == key:
                found_idx = idx
                break
            idx = (idx + 1) & (CAP - 1)

        if found_idx == -1:
            print(f"WARNING: Key {key} not found!")
        else:
            key_probes[key] = probes

    # Separate keys by probe distance
    easy_keys = [k for k, p in key_probes.items() if p < 10]
    hard_keys = [k for k, p in key_probes.items() if p >= 10]

    print(f"Keys with probe distance < 10: {len(easy_keys)}")
    print(f"Keys with probe distance >= 10: {len(hard_keys)}")

    # Shuffle easy keys and take first NUM_QUERIES
    random.shuffle(easy_keys)

    for i in range(NUM_QUERIES):
        key = easy_keys[i % len(easy_keys)]
        query_keys.append(key)
        expected_values.append(key_to_value[key])

    # Shuffle queries to randomize order
    combined = list(zip(query_keys, expected_values))
    random.shuffle(combined)
    query_keys, expected_values = zip(*combined)
    query_keys = list(query_keys)
    expected_values = list(expected_values)

    def u64_to_i64(u):
        """Convert unsigned 64-bit to signed int64 for struct packing."""
        if u >= (1 << 63):
            return u - (1 << 64)
        return u

    # Write simple_inserted_slot.data (hashtable)
    output_dir = "data_obj"
    with open(f"{output_dir}/simple_inserted_slot.data", "wb") as f:
        for key, value, padding in table:
            # Pack as: key(int64), value(int32), padding(int32)
            f.write(struct.pack("<q", u64_to_i64(key)))   # little-endian int64
            f.write(struct.pack("<i", value))  # little-endian int32
            f.write(struct.pack("<i", padding))  # little-endian int32

    # Write simple_lookup_keys.data
    with open(f"{output_dir}/simple_lookup_keys.data", "wb") as f:
        for key in query_keys:
            f.write(struct.pack("<q", u64_to_i64(key)))

    # Write simple_lookup_values.data
    with open(f"{output_dir}/simple_lookup_values.data", "wb") as f:
        for val in expected_values:
            f.write(struct.pack("<i", val))

    print(f"\nGenerated files in {output_dir}/:")
    print(f"  simple_inserted_slot.data:  {CAP * 16} bytes ({CAP} entries)")
    print(f"  simple_lookup_keys.data:    {NUM_QUERIES * 8} bytes ({NUM_QUERIES} keys)")
    print(f"  simple_lookup_values.data:  {NUM_QUERIES * 4} bytes ({NUM_QUERIES} values)")

    # Verify offsets exceed uint16_t
    max_offset = CAP * 16 - 8  # Last key starts 8 bytes before end
    print(f"\nMax byte offset: {max_offset} (uint16_t max: 65535)")
    if max_offset > 65535:
        print("  -> Exceeds uint16_t limit! Offset truncation will occur.")

    # Verify some sample hashes
    print("\nSample hash verification:")
    for i in range(min(5, len(query_keys))):
        key = query_keys[i]
        h = murmurhash3(key)
        idx = h & (CAP - 1)
        print(f"  key={key:20d} -> hash=0x{h:08x} -> idx={idx}")


if __name__ == "__main__":
    main()