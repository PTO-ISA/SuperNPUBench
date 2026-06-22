#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "template_asm.h"
#ifndef __linx
#include <stdio.h>
#endif

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
// hashfind — SIMT linear-probing hash table lookup kernel (no probe buffers)
//   Single kernel launch using 3D grid: <<<thread_num, 1, 1>>>
//   Each thread processes multiple keys with stride = thread_num
// ============================================================================

template <int32_t thread_num>
void __mtc__ hashfind(uint8_t* __restrict__ slot,
                      int64_t* __restrict__ keys,
                      int32_t* __restrict__ values_output,
                      uint32_t capacity,
                      int32_t num,
                      int32_t entry_size,
                      int32_t max_probe,
                      int64_t key_sentinel,
                      int32_t value_sentinel)
{
    int32_t tid = blkv_get_index_x();

    // main loop: each thread processes multiple keys with stride = thread_num
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

        uint32_t curr_slot = h % capacity;

        for (int probe_cnt = 0; probe_cnt < max_probe; probe_cnt++) {
            int64_t* key_addr = (int64_t*)(slot + entry_size * curr_slot);
            int64_t curr_key = *key_addr;

            if (curr_key == key_sentinel) {
                // key not found
                values_output[idx] = value_sentinel;
                probe_cnt = max_probe;
            }

            if (curr_key == key) {
                // found key
                int32_t* value_addr = (int32_t*)(slot + entry_size * curr_slot + sizeof(int64_t));
                values_output[idx] = value_addr[0];
                probe_cnt = max_probe;
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

    int64_t key_sentinel = int64_t(-1);
    int32_t value_sentinel = int32_t(-1);

    // Single kernel launch: each thread loops over multiple keys with stride = kNumThreads
    BENCHSTART;
    hashfind<kNumThreads><<<kNumThreads, 1, 1>>>(
        slots, keys, output,
        kCapacity, kNum,
        kEntrySize, kMaxProbe,
        key_sentinel, value_sentinel);
    BENCHEND;

#ifndef FOR_GFSIM
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

#ifndef __linx
    if (mismatch_count > 0) {
        printf("\n=== Mismatching keys (%d total) ===\n", mismatch_count);
        printf("%7s  %22s  %10s  %10s\n", "Idx", "Key", "Got", "Expected");
        for (int i = 0; i < kNum; i++) {
            if (output[i] != expected_vals[i]) {
                printf("%7d  %22lld  %10d  %10d\n",
                       i, (long long)keys[i], (int)output[i], (int)expected_vals[i]);
            }
        }
    }

    printf("\n=== hashtable_lookup_simt ===\n");
    printf("Match: %d/%d (%d %%)\n", match, kNum, int(100 * double(match) / double(kNum)));
    fflush(stdout);
#endif
    return (match == kNum) ? 0 : 1;
#else
    return 0;
#endif
}
