#!/usr/bin/env python3
"""Compute hash indices and offsets for first 256 query keys."""

import struct

# Read first 256 query keys
with open('/remote/lms01/j00827727/jcore/JanusCoreBench/test/kernel/control/hashtable_lookup_simd/data_obj/lookup_keys.data', 'rb') as f:
    keys = []
    for i in range(256):
        data = f.read(8)
        key = struct.unpack('<q', data)[0]
        keys.append(key)

# MurmurHash3 function (matching ACCEL reference)
def murmurhash3(low, high):
    k1 = low
    h = 0

    # Block 1: low
    k1 = (k1 * 0xcc9e2d51) & 0xFFFFFFFF
    k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
    k1 = (k1 * 0x1b873593) & 0xFFFFFFFF
    h ^= k1
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Block 2: high
    k1 = high
    k1 = (k1 * 0xcc9e2d51) & 0xFFFFFFFF
    k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
    k1 = (k1 * 0x1b873593) & 0xFFFFFFFF
    h ^= k1
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Finalization
    h = (h + 8) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16

    return h

mask = 2550000 - 1  # 2549999

# Compute value offsets (hash_idx * 16 + 8)
value_offsets = []
for i, key in enumerate(keys):
    low = key & 0xFFFFFFFF
    high = (key >> 32) & 0xFFFFFFFF
    h = murmurhash3(low, high)
    idx = h & mask
    offset = idx * 16 + 8  # value offset from table base
    value_offsets.append(offset)

# Find minimum
min_offset = min(value_offsets)
print(f"Min offset: {min_offset} (0x{min_offset:x})")

# Compute relative offsets
print(f"\nFirst 32 relative offsets (value):")
for i in range(32):
    rel_offset = value_offsets[i] - min_offset
    print(f"  [{i:2d}]: idx={value_offsets[i]//16}, abs_offset={value_offsets[i]}, rel_offset={rel_offset} (0x{rel_offset:04x})")

# Also compute key offsets (for comparison)
key_offsets = []
for i, key in enumerate(keys):
    low = key & 0xFFFFFFFF
    high = (key >> 32) & 0xFFFFFFFF
    h = murmurhash3(low, high)
    idx = h & mask
    offset = idx * 16 + 0  # key offset from table base
    key_offsets.append(offset)

# Generate C array for key offsets (relative to min)
print(f"\n// Key offsets array (relative to min_offset={min_offset})")
print("static const uint16_t g_key_offsets[256] = {")
for i in range(0, 256, 16):
    row = key_offsets[i:i+16]
    rel = [f"{x - min_offset:5d}" for x in row]
    print(f"    {', '.join(rel)},")
print("};")

# Generate C array for value offsets
print(f"\n// Value offsets array (relative to min_offset={min_offset})")
print("static const uint16_t g_val_offsets[256] = {")
for i in range(0, 256, 16):
    row = value_offsets[i:i+16]
    rel = [f"{x - min_offset:5d}" for x in row]
    print(f"    {', '.join(rel)},")
print("};")

print(f"\n// Min offset constant")
print(f"static const uint32_t g_min_offset = {min_offset};")