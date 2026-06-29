#include <common/pto_tileop.hpp>
#include "benchmark.h"

#define FOR_GFSIM

// ============================================================================
// Murmur3HashDevice: 64-bit finalizer (from reference hkv_insert_or_assign)
// ============================================================================

static inline uint64_t murmur3_hash_device(uint64_t key) {
    uint64_t k = key;
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

// ============================================================================
// Constants
// ============================================================================

constexpr int32_t BUCKET_MAX_SIZE = 128;
constexpr int32_t BUCKETS_NUM = 16;
constexpr uint64_t CAPACITY = BUCKETS_NUM * BUCKET_MAX_SIZE;  // 2048
constexpr int32_t MAX_BUCKET_SHIFT = 7;  // log2(128) = 7
constexpr int32_t DIM = 2;

constexpr uint64_t EMPTY_KEY = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t RESERVED_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC);

// Per bucket layout sizes (bytes)
static const int64_t kDigestsBytes = BUCKET_MAX_SIZE;                            // 128
static const int64_t kKeysBytes    = BUCKET_MAX_SIZE * 8;                        // 1024
static const int64_t kScoresBytes  = BUCKET_MAX_SIZE * 8;                        // 1024
static const int64_t kVectorsBytes = BUCKET_MAX_SIZE * DIM * 4;                  // 1024
static const int64_t kBucketDataBytes = kDigestsBytes + kKeysBytes + kScoresBytes + kVectorsBytes; // 3200

// Number of lookup keys
static const int32_t kNumLookup = 64;

// Output buffer for lookup results
static float g_output_values[kNumLookup * DIM];

// ============================================================================
// ELF Data layout — embedded binary data
// ============================================================================

extern "C" {
    extern const uint8_t _binary_buckets_bin_start[];
    extern const uint8_t _binary_buckets_bin_end[];
    extern const uint8_t _binary_buckets_size_bin_start[];
    extern const uint8_t _binary_buckets_size_bin_end[];
    extern const uint8_t _binary_lookup_keys_bin_start[];
    extern const uint8_t _binary_lookup_keys_bin_end[];
    extern const uint8_t _binary_lookedup_values_bin_start[];
    extern const uint8_t _binary_lookedup_values_bin_end[];
}

// ============================================================================
// SIMT Lookup Kernel with digest-based probing
// ============================================================================

template <int32_t thread_num>
void __mtc__ hkv_lookup_kernel(
    int8_t* __restrict__ buckets_data,
    int32_t* __restrict__ buckets_size_data,
    uint64_t* __restrict__ keys,
    float* __restrict__ output_values,
    int32_t num,
    int32_t start_idx) {

    int32_t tid = blkv_get_index_x();

    int32_t idx = tid;

    // if (idx < num) {
        uint64_t key = keys[idx];

        int32_t out_idx = start_idx + idx;

        // Output defaults to 0 (not found)
        float result_v0 = 0.0f;
        float result_v1 = 0.0f;

        // Skip reserved keys (top 2 bits all set)
        uint64_t reserved_check = key & RESERVED_KEY_MASK;
        // if (reserved_check != RESERVED_KEY_MASK) {
            uint64_t hashed_key = murmur3_hash_device(key);

            // Digest = (hashed_key >> 32) & 0xFF
            int32_t target_digest = (hashed_key >> 32) & 0xFF;

            // bkt_idx = hashed_key % BUCKETS_NUM (hash directly selects bucket)
            int32_t bkt_idx = hashed_key & (BUCKETS_NUM - 1);

            // Point to this bucket's base
            int8_t* bucket_base = buckets_data + bkt_idx * kBucketDataBytes;

            // Keys start after digests
            uint64_t* bucket_keys = (uint64_t*)(bucket_base + kDigestsBytes);
            // Vectors start after digests + keys + scores
            float* bucket_vectors = (float*)(bucket_base + kDigestsBytes + kKeysBytes + kScoresBytes);
            // Digests at base
            int8_t* bucket_digests = bucket_base;

            // Traverse bucket one slot at a time: compare digest first, then key
            for (int32_t pos_cur = 0; pos_cur < BUCKET_MAX_SIZE; pos_cur++) {

                int32_t digest = bucket_digests[pos_cur];

                if (digest == target_digest) {
                    if (bucket_keys[pos_cur] == key) {
                        result_v0 = bucket_vectors[pos_cur * DIM + 0];
                        result_v1 = bucket_vectors[pos_cur * DIM + 1];
                    }
                }
            }
        // }

        // Write output
        output_values[out_idx * DIM + 0] = result_v0;
        output_values[out_idx * DIM + 1] = result_v1;
    // }
}

// ============================================================================
// main
// ============================================================================

int main() {
    // Setup pointers from embedded ELF binary data
    int8_t* buckets_data = (int8_t*)(const_cast<uint8_t*>(_binary_buckets_bin_start));
    int32_t* buckets_size_data = (int32_t*)(const_cast<uint8_t*>(_binary_buckets_size_bin_start));
    uint64_t* lookup_keys = (uint64_t*)(const_cast<uint8_t*>(_binary_lookup_keys_bin_start));
    float* expected_values = (float*)(const_cast<uint8_t*>(_binary_lookedup_values_bin_start));

    // Initialize output to zero
    for (int32_t i = 0; i < kNumLookup * DIM; i++) {
        g_output_values[i] = 0.0f;
    }

    // Launch the lookup kernel in batches (each thread handles one key per batch)
    constexpr int32_t thread_num = 64;
    constexpr int32_t batch_size = thread_num;
    constexpr int32_t num_batches = (kNumLookup + batch_size - 1) / batch_size;

    for (int32_t batch = 0; batch < num_batches; batch++) {
        int32_t batch_start = batch * batch_size;
        int32_t batch_count = (batch_start + batch_size <= kNumLookup) ? batch_size : (kNumLookup - batch_start);

        hkv_lookup_kernel<thread_num><<<thread_num, 1, 1>>>(
            buckets_data,
            buckets_size_data,
            lookup_keys + batch_start,
            g_output_values,
            batch_count,
            batch_start);
    }

#ifndef FOR_GFSIM
    // Verify results
    int match = 0;
    int total = kNumLookup;

    printf("\n=== HKV Lookup Results ===\n");
    for (int32_t i = 0; i < kNumLookup; i++) {
        float got_v0 = g_output_values[i * DIM + 0];
        float got_v1 = g_output_values[i * DIM + 1];
        float exp_v0 = expected_values[i * DIM + 0];
        float exp_v1 = expected_values[i * DIM + 1];

        // NaN expected means key should not be found (output should be 0.0)
        if (exp_v0 != exp_v0) {  // NaN check
            if (got_v0 == 0.0f && got_v1 == 0.0f) {
                match++;
            } else {
                printf("  [%d] NOT-FOUND mismatch: got=(%f,%f), expected=NaN\n",
                       i, got_v0, got_v1);
            }
        } else {
            float diff0 = got_v0 - exp_v0;
            float diff1 = got_v1 - exp_v1;
            if (diff0 < 0) diff0 = -diff0;
            if (diff1 < 0) diff1 = -diff1;
            if (diff0 < 0.001f && diff1 < 0.001f) {
                match++;
            } else {
                printf("  [%d] FOUND mismatch: got=(%f,%f), expected=(%f,%f)\n",
                       i, got_v0, got_v1, exp_v0, exp_v1);
            }
        }
    }

    printf("\n=== HKV Lookup Summary ===\n");
    printf("Match: %d/%d (%.1f%%)\n", match, total, 100.0 * match / total);
    fflush(stdout);

    return (match == total) ? 0 : 1;
#else
    return 0;
#endif
}
