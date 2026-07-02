#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "template_asm.h"
#include <stdio.h>
#include <string.h>

#include "sort/topk.hpp"

// #define FOR_GFSIM
// ============================================================================
// ELF Data layout
// ============================================================================

extern "C" {
    extern const uint8_t _binary_input_131072_data_start[];
    extern const uint8_t _binary_input_131072_data_end[];
    extern const uint8_t _binary_top_2048_out_data_start[];
    extern const uint8_t _binary_top_2048_out_data_end[];
}

static uint16_t* g_input    = reinterpret_cast<uint16_t*>(
    const_cast<uint8_t*>(_binary_input_131072_data_start));
static uint16_t* g_expected = reinterpret_cast<uint16_t*>(
    const_cast<uint8_t*>(_binary_top_2048_out_data_start));

// ============================================================================
// Global-scope buffers
// ============================================================================

static uint16_t g_output[kInputCount];

// ============================================================================
// main
// ============================================================================

int main() {
#ifndef FOR_GFSIM
    printf("=== TopK Test (SIMT per-bucket) ===\n");
    printf("Input: %d  TopK: %d  Tiles: %d  TileSize: %d\n",
           kInputCount, kTopK, kNumTiles, kTileSize);
    fflush(stdout);
#endif

    // -------------------------------------------------------------------------
    // Phase 1: SIMT high8 histogram (1 block × 256 lanes, each lane = 1 bucket)
    // -------------------------------------------------------------------------
    TileU32 high8HistTile;
    TEXPANDSCALAR(high8HistTile, static_cast<uint32_t>(0));
    ExtractHigh8Hist_Impl< TileU32 >(high8HistTile, g_input);

    // Copy histogram results out and reduce to global 256-bin histogram
    using HistGT = GlobalTensor<uint32_t, Shape<1,1,1,16,16>, Stride<1,1,1,16,1>>;
    uint32_t histResult[256];
    HistGT histGlobal(histResult);
    TCOPYOUT(histGlobal, high8HistTile);

    uint32_t global_high8_hist[256] = {0};
    for (int b = 0; b < 256; b++) {
        global_high8_hist[b] = histResult[b];
    }

#ifndef FOR_GFSIM
    printf("\nPhase 1: high8 histograms built (1 SIMT launch, 256 lanes).\n");
    fflush(stdout);
#endif

    // -------------------------------------------------------------------------
    // Phase 2: Scalar prefix scan → kth_bin and need_from_kth_bin
    // -------------------------------------------------------------------------
    int need_from_kth_bin = 0;
    int kth_bin = find_kth_bin(global_high8_hist, kTopK, need_from_kth_bin);

#ifndef FOR_GFSIM
    printf("\nPhase 2: kth_bin=%d  need_from_kth_bin=%d\n",
           kth_bin, need_from_kth_bin);
    uint64_t total_above = 0;
    for (int b = kth_bin + 1; b < 256; b++) total_above += global_high8_hist[b];
    printf("  Elements in bins > kth_bin: %lu  (expected ~%d)\n",
           total_above, kTopK - need_from_kth_bin);
    printf("  Elements in bin == kth_bin:  %u\n", global_high8_hist[kth_bin]);
    fflush(stdout);
#endif

    // -------------------------------------------------------------------------
    // Phase 3: SIMT low8 histogram for kth_bin elements
    // -------------------------------------------------------------------------
    TileU32 low8HistTile;
    TEXPANDSCALAR(low8HistTile, static_cast<uint32_t>(0));
    ExtractLow8HistForKthBin_Impl< TileU32 >(low8HistTile, g_input,
                                             static_cast<uint16_t>(kth_bin));

    uint32_t low8HistResult[256];
    HistGT low8HistGlobal(low8HistResult);
    TCOPYOUT(low8HistGlobal, low8HistTile);

    uint32_t global_low8_hist_kth[256] = {0};
    for (int b = 0; b < 256; b++) {
        global_low8_hist_kth[b] = low8HistResult[b];
    }

    // -------------------------------------------------------------------------
    // Phase 4: Scalar prefix scan → low8_boundary
    // -------------------------------------------------------------------------
    int low8_boundary = 0;
    uint64_t cumsum_low = 0;
    for (int b = 255; b >= 0; b--) {
        cumsum_low += global_low8_hist_kth[b];
        if (cumsum_low >= static_cast<uint64_t>(need_from_kth_bin)) {
            low8_boundary = b;
            break;
        }
    }

#ifndef FOR_GFSIM
    printf("\nPhase 4: low8_boundary=%d\n", low8_boundary);
    printf("  Global low8 hist (kth bin) total: %lu\n", cumsum_low);
    fflush(stdout);
#endif

    // -------------------------------------------------------------------------
    // Phase 5: Scalar masked scatter (directly on g_input / g_output)
    // -------------------------------------------------------------------------
    memset(g_output, 0, sizeof(g_output));
    for (int i = 0; i < kInputCount; i++) {
        uint16_t val  = g_input[i];
        uint8_t  high8 = static_cast<uint8_t>(val >> 8);
        int      low8  = static_cast<int>(val & 0xFF);
        int include = (high8 > kth_bin) ||
                      ((high8 == kth_bin) & (low8 >= low8_boundary));
        if (include) {
            g_output[i] = val;
        }
    }

    // -------------------------------------------------------------------------
    // Host: collect non-zero entries
    // -------------------------------------------------------------------------
    uint16_t result[kTopK];
    int out_count = 0;
    for (int i = 0; i < kInputCount && out_count < kTopK; i++) {
        if (g_output[i] != 0) {
            result[out_count++] = g_output[i];
        }
    }

#ifndef FOR_GFSIM
    printf("\nPhase 5: Collected %d output elements (expected %d)\n",
           out_count, kTopK);
    fflush(stdout);
#endif

    // -------------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------------
    int cmp_count = (out_count < kTopK) ? out_count : kTopK;

    uint16_t result_sorted[2048];
    memcpy(result_sorted, result, sizeof(result_sorted));
    for (int i = 0; i < cmp_count; i++) {
        for (int j = i + 1; j < cmp_count; j++) {
            if (result_sorted[i] < result_sorted[j]) {
                uint16_t tmp = result_sorted[i];
                result_sorted[i] = result_sorted[j];
                result_sorted[j] = tmp;
            }
        }
    }

    int match = 0;
    for (int i = 0; i < cmp_count; i++) {
        if (result_sorted[i] == g_expected[i]) match++;
    }

#ifndef FOR_GFSIM
    printf("\n=== Verification (vs. embedded standard answer) ===\n");
    printf("Match: %d/%d (%.1f%%)\n", match, cmp_count, 100.0 * match / cmp_count);
    printf("Output[0..9]:    ");
    for (int i = 0; i < 10 && i < out_count; i++) printf("0x%04x ", result_sorted[i]);
    printf("\nExpected[0..9]:  ");
    for (int i = 0; i < 10; i++) printf("0x%04x ", g_expected[i]);
    printf("\n");
#endif

    int ret = (match == cmp_count) ? 0 : 1;
#ifndef FOR_GFSIM
    printf("%s\n", ret ? "FAIL" : "PASS");
    fflush(stdout);
#endif
    return ret;
}
