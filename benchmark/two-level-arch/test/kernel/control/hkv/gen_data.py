#!/usr/bin/env python3
"""
Generate HKV (Hash Key-Value) test dataset for lookup kernel testing.
- 16 buckets, bucket_capacity = 128
- Key type: uint64_t, Score type: uint64_t, Digest type: uint8_t
- Value type: float (dim=2, stored as uint2/byte8 for vec2 writes)

Data layout per bucket (matching Bucket struct in reference):
  digests_: 128 bytes (uint8_t[128])  — stored BEFORE keys
  keys_:    128 * 8 = 1024 bytes (uint64_t[128])
  scores_:  128 * 8 = 1024 bytes (uint64_t[128])
  vectors:  128 * 2 * 4 = 1024 bytes (float[128][2]) — dim=2, vec2

Bucket memory layout (from BUCKET::digests helper):
  digests are at keys_ptr - bucket_capacity + offset (i.e. before keys)
  scores are at keys_ptr + bucket_capacity + offset (i.e. after keys)

For this test we store:
  buckets.bin    — all bucket data concatenated (digests + keys + scores + vectors for each bucket)
  buckets_size.bin — int32_t[16], the number of occupied slots per bucket
  lookup_keys.bin  — uint64_t keys to look up
  lookedup_values.bin — expected lookup results (float pairs, or sentinel for not-found)

Murmur3HashDevice (64-bit finalizer, from reference):
  k ^= k >> 33
  k *= 0xff51afd7ed558ccd
  k ^= k >> 33
  k *= 0xc4ceb9fe1a85ec53
  k ^= k >> 33
  return k

Digest = (hashed_key >> 32) & 0xFF
"""

import struct
import random

EMPTY_KEY = 0xFFFFFFFFFFFFFFFF
BUCKETS_NUM = 16
BUCKET_CAPACITY = 128
DIM = 2  # float pairs per value

def murmur3_hash_device(key):
    """64-bit MurmurHash3 finalizer matching the device code."""
    k = key & 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    k = (k * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    k = (k * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    return k

def get_digest(key):
    h = murmur3_hash_device(key)
    return (h >> 32) & 0xFF

def empty_digest():
    h = murmur3_hash_device(EMPTY_KEY)
    return (h >> 32) & 0xFF

def get_global_idx(hashed_key, capacity):
    """Simplified: global_idx = hashed_key % capacity (capacity = BUCKETS_NUM * BUCKET_CAPACITY)"""
    return hashed_key % capacity

def main():
    random.seed(42)

    capacity = BUCKETS_NUM * BUCKET_CAPACITY  # 2048 total slots
    max_bucket_shift = 7  # log2(BUCKET_CAPACITY) = log2(128) = 7

    # Generate random keys (not reserved)
    NUM_KEYS = int(capacity * 0.5)  # 50% load factor = 1024 keys
    NUM_LOOKUP = 256  # number of lookup queries

    all_keys = set()
    keys_list = []
    while len(keys_list) < NUM_KEYS:
        key = random.getrandbits(64)
        # Skip reserved keys (top 2 bits set)
        if (key & 0xFFFFFFFFFFFFFFFC) == 0xFFFFFFFFFFFFFFFC:
            continue
        if key not in all_keys:
            all_keys.add(key)
            keys_list.append(key)

    # Build buckets
    # Each bucket: digests[128], keys[128], scores[128], vectors[128*2]
    buckets_digests = []
    buckets_keys = []
    buckets_scores = []
    buckets_vectors = []
    buckets_size = [0] * BUCKETS_NUM

    for b in range(BUCKETS_NUM):
        ed = empty_digest()
        digests = [ed] * BUCKET_CAPACITY
        keys = [EMPTY_KEY] * BUCKET_CAPACITY
        scores = [0] * BUCKET_CAPACITY  # uint64_t
        vectors = [0.0] * (BUCKET_CAPACITY * DIM)
        buckets_digests.append(digests)
        buckets_keys.append(keys)
        buckets_scores.append(scores)
        buckets_vectors.append(vectors)

    # Key -> (bucket_idx, slot_idx, value) mapping
    key_to_value = {}

    # Insert keys into buckets
    for i, key in enumerate(keys_list):
        hashed_key = murmur3_hash_device(key)
        global_idx = hashed_key % capacity
        bkt_idx = global_idx >> max_bucket_shift
        slot_idx = global_idx & (BUCKET_CAPACITY - 1)

        # Linear probing within bucket
        inserted = False
        for probe in range(BUCKET_CAPACITY):
            pos = (slot_idx + probe) & (BUCKET_CAPACITY - 1)
            if buckets_keys[bkt_idx][pos] == EMPTY_KEY:
                digest = get_digest(key)
                buckets_digests[bkt_idx][pos] = digest
                buckets_keys[bkt_idx][pos] = key
                buckets_scores[bkt_idx][pos] = i + 1  # score = index+1
                # Value: two floats
                v0 = float(i + 1) * 1.0
                v1 = float(i + 1) * 2.0
                buckets_vectors[bkt_idx][pos * DIM + 0] = v0
                buckets_vectors[bkt_idx][pos * DIM + 1] = v1
                buckets_size[bkt_idx] += 1
                key_to_value[key] = (v0, v1)
                inserted = True
                break
        if not inserted:
            print(f"WARNING: Could not insert key {key} into bucket {bkt_idx}")

    print(f"Inserted {NUM_KEYS} keys into {BUCKETS_NUM} buckets")
    for b in range(BUCKETS_NUM):
        print(f"  Bucket {b}: {buckets_size[b]}/{BUCKET_CAPACITY} occupied")

    # Generate lookup keys: 80% found, 20% not found
    found_keys = list(keys_list[:NUM_LOOKUP * 4 // 5])
    # Generate not-found keys
    not_found_keys = []
    while len(not_found_keys) < NUM_LOOKUP - len(found_keys):
        key = random.getrandbits(64)
        if (key & 0xFFFFFFFFFFFFFFFC) == 0xFFFFFFFFFFFFFFFC:
            continue
        if key not in all_keys:
            not_found_keys.append(key)

    lookup_keys = found_keys + not_found_keys
    random.shuffle(lookup_keys)

    # Compute expected lookup results
    # For found keys: (v0, v1) pair; for not-found: (NaN, NaN) sentinel
    NOT_FOUND_V0 = float('nan')
    NOT_FOUND_V1 = float('nan')
    lookedup_values = []
    for key in lookup_keys:
        if key in key_to_value:
            lookedup_values.append(key_to_value[key])
        else:
            lookedup_values.append((NOT_FOUND_V0, NOT_FOUND_V1))

    # Write binary files
    output_dir = "data_obj"

    # 1. buckets.bin — per bucket: digests[128] + keys[128] + scores[128] + vectors[128*2]
    with open(f"{output_dir}/buckets.bin", "wb") as f:
        for b in range(BUCKETS_NUM):
            # digests: 128 bytes
            for d in buckets_digests[b]:
                f.write(struct.pack("<B", d))
            # keys: 128 * 8 bytes
            for k in buckets_keys[b]:
                f.write(struct.pack("<Q", k))
            # scores: 128 * 8 bytes
            for s in buckets_scores[b]:
                f.write(struct.pack("<Q", s))
            # vectors: 128 * 2 * 4 bytes (float32)
            for v in buckets_vectors[b]:
                f.write(struct.pack("<f", v))

    # 2. buckets_size.bin — int32_t[16]
    with open(f"{output_dir}/buckets_size.bin", "wb") as f:
        for s in buckets_size:
            f.write(struct.pack("<i", s))

    # 3. lookup_keys.bin — uint64_t[NUM_LOOKUP]
    with open(f"{output_dir}/lookup_keys.bin", "wb") as f:
        for key in lookup_keys:
            f.write(struct.pack("<Q", key))

    # 4. lookedup_values.bin — float[NUM_LOOKUP * 2]
    with open(f"{output_dir}/lookedup_values.bin", "wb") as f:
        for v0, v1 in lookedup_values:
            f.write(struct.pack("<f", v0))
            f.write(struct.pack("<f", v1))

    # 5. key_score_digest.bin — reference: all inserted keys with their scores and digests
    #    Format: key(uint64) + score(uint64) + digest(uint8) per entry, for all NUM_KEYS entries
    with open(f"{output_dir}/key_score_digest.bin", "wb") as f:
        for i, key in enumerate(keys_list):
            digest = get_digest(key)
            score = i + 1
            f.write(struct.pack("<Q", key))
            f.write(struct.pack("<Q", score))
            f.write(struct.pack("<B", digest))

    print(f"\nGenerated files in {output_dir}/:")
    print(f"  buckets.bin:         {BUCKETS_NUM * (128 + 128*8 + 128*8 + 128*2*4)} bytes")
    print(f"  buckets_size.bin:    {BUCKETS_NUM * 4} bytes")
    print(f"  lookup_keys.bin:     {NUM_LOOKUP * 8} bytes")
    print(f"  lookedup_values.bin: {NUM_LOOKUP * 2 * 4} bytes")
    print(f"  key_score_digest.bin:{NUM_KEYS * (8 + 8 + 1)} bytes")

    found_count = sum(1 for v in lookedup_values if v[0] == v[0])  # NaN != NaN
    print(f"\nLookup keys: {NUM_LOOKUP} ({found_count} found, {NUM_LOOKUP - found_count} not found)")

if __name__ == "__main__":
    main()
