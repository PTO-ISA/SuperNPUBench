#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "template_asm.h"
#include <stdio.h>

#include "control/hashtable_lookup_simd.hpp"

// ============================================================================
// Constants
// ============================================================================

constexpr uint32_t kCapacity   = 2550000;  // 40800000 / 16
#ifndef kNum
constexpr int32_t  kNum        = 256;      // 3276800 / 8
#endif

#ifndef MAX_PROBE
#define MAX_PROBE 512
#endif

#ifndef NUM_COL
#define NUM_COL 256
#endif

constexpr int32_t  kMaxProbe   = MAX_PROBE;
constexpr int32_t  kNumRow   = 1;
constexpr int32_t  kNumCol   = NUM_COL;

constexpr int32_t  kBatchSize  = kNumCol;      // 1D tile of size NUM_COL

// ============================================================================
// ELF Data layout
// ============================================================================

extern "C" {
    extern const uint8_t _binary_inserted_slot_data_start[];
    extern const uint8_t _binary_inserted_slot_data_end[];
    extern const uint8_t _binary_lookup_keys_data_start[];
    extern const uint8_t _binary_lookup_keys_data_end[];
    extern const uint8_t _binary_lookup_values_data_start[];
    extern const uint8_t _binary_lookup_values_data_end[];
}

static TableEntry* g_hashtable_ro = reinterpret_cast<TableEntry*>(
    const_cast<uint8_t*>(_binary_inserted_slot_data_start));
static int64_t* g_query_keys = reinterpret_cast<int64_t*>(
    const_cast<uint8_t*>(_binary_lookup_keys_data_start));
static int32_t* g_lookup_values = reinterpret_cast<int32_t*>(
    const_cast<uint8_t*>(_binary_lookup_values_data_start));

static int32_t g_output[kNum];

// ============================================================================
// main
// ============================================================================

int main() {
    // Initialize all output to kNotFound
    for (int i = 0; i < kNum; i++) {
        g_output[i] = kNotFound;
    }

    // Process all lookup keys in batches of kBatchSize (256)
    constexpr int32_t num_batches = (kNum + kBatchSize - 1) / kBatchSize;

    BENCHSTART;
    for (int32_t batch = 0; batch < num_batches; batch++) {
        int32_t batch_start = batch * kBatchSize;
        int32_t batch_count = (batch_start + kBatchSize <= kNum) ? kBatchSize : (kNum - batch_start);

        // For the last batch that may be smaller, we still launch with full tile
        // but only the valid elements matter
        LaunchHashFind<kNumRow, kNumCol, kCapacity, kMaxProbe>(
            g_output + batch_start, g_hashtable_ro, g_query_keys + batch_start);
    }
    BENCHEND;

#ifndef FOR_GFSIM
    // Verify results
    int match = 0;
    int mismatch_count = 0;
    for (int i = 0; i < kNum; i++) {
        if (g_output[i] == g_lookup_values[i]) {
            match++;
        } else {
            mismatch_count++;
        }
    }

    printf("=== hashtable_lookup_simd ===\n");
    printf("Match: %d/%d (%.4f%%)\n", match, kNum, 100.0 * double(match) / double(kNum));

    if (mismatch_count > 0) {
        printf("\n=== First 20 mismatches ===\n");
        printf("%7s  %22s  %10s  %10s\n", "Idx", "Key", "Got", "Expected");
        int shown = 0;
        for (int i = 0; i < kNum && shown < 20; i++) {
            if (g_output[i] != g_lookup_values[i]) {
                printf("%7d  %22lld  %10d  %10d\n",
                       i, (long long)g_query_keys[i], (int)g_output[i], (int)g_lookup_values[i]);
                shown++;
            }
        }
    }
    fflush(stdout);
#endif

#ifndef FOR_GFSIM
    int ret = (match == kNum) ? 0 : 1;
    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    fflush(stdout);
    return ret;
#else
    return 0;
#endif
}
