#include <common/pto_tileop.hpp>
#include "benchmark.h"

// ============================================================================
// ELF Data layout — embedded binary data produced by build_data_obj.sh
// ============================================================================

extern "C" {
    extern const uint8_t _binary_inserted_slot_data_start[];
    extern const uint8_t _binary_inserted_slot_data_end[];
    extern const uint8_t _binary_lookup_keys_data_start[];
    extern const uint8_t _binary_lookup_keys_data_end[];
    extern const uint8_t _binary_lookup_values_data_start[];
    extern const uint8_t _binary_lookup_values_data_end[];
}

// ============================================================================
// Constants
// ============================================================================

constexpr uint32_t kCapacity    = 2550000;   // 40800000 / 16 = 2550000 entries
#ifndef kNum
constexpr int32_t  kNum         = 409600;    // 3276800 / 8 = 409600 query keys
#endif

#ifndef kNumThreads
constexpr int32_t  kNumThreads         = 1024;    // 3276800 / 8 = 409600 query keys
#endif

constexpr int32_t  kEntrySize   = 16;     // 8-byte key + 4-byte value + 4-byte padding

#ifndef MAX_PROBE
#define MAX_PROBE 512
#endif

constexpr int32_t  kMaxProbe    = MAX_PROBE;

// ============================================================================
// MurmurHash3 constants
// ============================================================================

#define DEFAULT_HASH_SEED 0u
#define c1 0xcc9e2d51u
#define c2 0x1b873593u
#define c3 0xe6546b64u
#define rot_c1 15u
#define rot_c2 13u

// ============================================================================
// Global buffer for SIMT kernel to write computed hash values
// ============================================================================
uint32_t g_hash_output[kNum];

// ============================================================================
// hashfind — SIMT linear-probing hash table lookup kernel (no probe buffers)
//   Single kernel launch with 1024 threads, each thread uses a for-loop
//   to process keys at stride thread_num:
//   tid, tid + thread_num, tid + 2*thread_num, ...
//
//   NOTE: Due to __mtc__ SIMT kernel limitations, the for-loop only executes
//   the first iteration. Only the first thread_num keys are correctly processed.
// ============================================================================

template <int32_t thread_num>
void __mtc__ hashfind(uint8_t* __restrict__ slot,
                      int64_t* __restrict__ keys,
                      int32_t* __restrict__ values_output,
                      uint32_t* __restrict__ hashes_output,
                      uint32_t capacity,
                      int32_t num,
                      int32_t entry_size,
                      int32_t max_probe)
{
    int32_t tid = blkv_get_index_x();
    // Each thread handles keys at stride thread_num: tid, tid+thread_num, tid+2*thread_num, ...
    for (int32_t idx = tid; idx < num; idx += thread_num)
    {
        int64_t key = keys[idx];

        uint32_t len = 8u;
        uint32_t* blocks = (uint32_t*)(keys + idx);

        uint32_t h = DEFAULT_HASH_SEED;
        for (int32_t j = 0; j < 2; j++)
        {
            uint32_t k1 = blocks[j];
            k1 *= c1;
            k1 = (k1 << rot_c1) | (k1 >> (32u - rot_c1));
            k1 *= c2;
            h ^= k1;
            h = (h << rot_c2) | (h >> (32u - rot_c2));
            h = h * 5u + c3;
        }

        // Finalize hash.
        h ^= len;
        h ^= h >> 16u;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        h *= 0xc2b2ae35u;
        h ^= h >> 16u;

        // Output hash value from SIMT kernel for host-side verification
        hashes_output[idx] = h;

        uint32_t curr_slot = h % capacity;

        for (int probe_cnt = 0; probe_cnt < max_probe; probe_cnt++) {
            int64_t* key_addr = (int64_t*)(slot + entry_size * curr_slot);
            int64_t curr_key = *key_addr;
            int32_t* value_addr = (int32_t*)(slot + entry_size * curr_slot + sizeof(int64_t));
            int32_t curr_val = value_addr[0];

            if (curr_key == key) {
                // found key
                values_output[idx] = curr_val;
            }
            curr_slot = (curr_slot + 1u) % capacity;
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    // Pointers initialized from embedded ELF binary data
    uint8_t*  slots          = const_cast<uint8_t*>(_binary_inserted_slot_data_start);
    int64_t*  keys           = reinterpret_cast<int64_t*>(const_cast<uint8_t*>(_binary_lookup_keys_data_start));
    int32_t*  expected_vals  = reinterpret_cast<int32_t*>(const_cast<uint8_t*>(_binary_lookup_values_data_start));

    static int32_t output[kNum];
    for (int i = 0; i < kNum; i++) output[i] = -1;

    // Single kernel launch: each thread processes keys at stride thread_num via for-loop
    BENCHSTART;
    hashfind<kNumThreads><<<kNumThreads, 1, 1>>>(
        slots, keys, output, g_hash_output,
        kCapacity, kNum,
        kEntrySize, kMaxProbe);
    BENCHEND;

#ifndef FOR_GFSIM
    // Print SIMT kernel computed hash values for first 64 keys
    printf("\n=== SIMT kernel hash values (first 64 keys) ===\n");
    printf("%4s  %22s  %10s  %7s  %10s  %10s\n", "Idx", "Key", "Hash(hex)", "Slot", "SIMT_out", "Expected");
    printf("%s\n", "------------------------------------------------------------------------------------");
    for (int i = 0; i < 64 && i < kNum; i++) {
        uint32_t h = g_hash_output[i];
        uint32_t slot = h % kCapacity;
        printf("%4d  %22lld  0x%08x  %7d  %10d  %10d\n",
               i, (long long)keys[i], h, slot, (int)output[i], (int)expected_vals[i]);
    }
#endif
    // Verify results
    int match = 0;
    int mismatch_count = 0;
    for (int i = 0; i < kNum; i++) {
        if (output[i] == expected_vals[i]) {
            match++;
        } else {
            mismatch_count++;
        }
    }

#ifndef FOR_GFSIM
    if (mismatch_count > 0) {
        printf("\n=== Mismatching keys (%d total) ===\n", mismatch_count);
        printf("%7s  %22s  %10s  %7s  %10s  %10s\n", "Idx", "Key", "Hash(hex)", "Slot", "Got", "Expected");
        for (int i = 0; i < kNum; i++) {
            if (output[i] != expected_vals[i]) {
                uint32_t h = g_hash_output[i];
                uint32_t slot = h % kCapacity;
                printf("%7d  %22lld  0x%08x  %7d  %10d  %10d\n",
                       i, (long long)keys[i], h, slot, (int)output[i], (int)expected_vals[i]);
            }
        }
    }

    printf("\n=== hashtable_lookup_simt_v2 ===\n");
    printf("Match: %d/%d (%.4f%%)\n", match, kNum, 100.0 * double(match) / double(kNum));
    fflush(stdout);
#endif
    return (match == kNum) ? 0 : 1;
}
