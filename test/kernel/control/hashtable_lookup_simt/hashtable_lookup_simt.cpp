#include <common/pto_tileop.hpp>
#include "benchmark.h"

#include "control/hashtable_lookup_simt.hpp"

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
    return (match == kNum) ? 0 : 1;
#else
    return 0;
#endif
}
