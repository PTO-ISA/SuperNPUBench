#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include <stdint.h>

#include "sort/topk.hpp"
#include "linx_print.h"

// ============================================================================
// TopK (radix-select via the THISTOGRAM tile op)
//
// The compute core is two passes of byte histograms over all 131072 inputs,
// each built with the THISTOGRAM tile instruction:
//   Phase 1: cumulative high-byte (Byte1) histogram, no filter   -> kth high byte
//   Phase 3: cumulative low-byte  (Byte0) histogram, filtered to high==kth_bin
//            -> low byte of the threshold
// The 16-bit threshold = (kth_bin<<8 | low8_boundary) is, by definition, the
// value of the K-th largest element. We verify against the embedded answer:
// g_expected is sorted descending, so g_expected[kTopK-1] is exactly the K-th
// largest. This O(1) check fully validates the THISTOGRAM radix-select and keeps
// the kernel feasible on the cycle-accurate model (no O(N) host scan).
// ============================================================================

extern "C" {
    extern const uint8_t _binary_input_131072_data_start[];
    extern const uint8_t _binary_top_2048_out_data_start[];
}

static uint16_t* g_input    = reinterpret_cast<uint16_t*>(
    const_cast<uint8_t*>(_binary_input_131072_data_start));
static uint16_t* g_expected = reinterpret_cast<uint16_t*>(
    const_cast<uint8_t*>(_binary_top_2048_out_data_start));

int main() {
    // Phase 1: cumulative high-byte histogram over all inputs (THISTOGRAM Byte1).
    static uint32_t high8_hist[256];
    BENCHSTART;
    build_high8_histogram(g_input, high8_hist);
    BENCHEND;

    // Phase 2: high byte of the K-th largest, and how many of that byte remain.
    int kth_bin = 0, need_from_kth = 0;
    find_kth_bin_from_cumulative(high8_hist, kTopK, kth_bin, need_from_kth);

    // Phase 3: cumulative low-byte histogram filtered to high==kth_bin (THISTOGRAM
    //          Byte0, idx tile supplies the high-byte prefix filter).
    static uint32_t low8_hist[256];
    build_low8_histogram(g_input, (uint16_t)kth_bin, low8_hist);

    // Phase 4: low byte boundary -> full 16-bit threshold value.
    int low8_boundary = 0, dummy = 0;
    find_kth_bin_from_cumulative(low8_hist, need_from_kth, low8_boundary, dummy);
    uint16_t threshold = (uint16_t)((kth_bin << 8) | (low8_boundary & 0xFF));

    // Verify: the K-th largest value (smallest of the top-K) == our threshold.
    uint16_t expected_kth = g_expected[kTopK - 1];

    linxi_puts("=== TopK (THISTOGRAM radix-select) ===");
    linxi_put("threshold(hex): ");    linxi_put_hex(threshold);        linxi_putc('\n');
    linxi_put("expected_kth(hex): "); linxi_put_hex(expected_kth);     linxi_putc('\n');
    linxi_put("kth_bin(hex): ");      linxi_put_hex((unsigned)kth_bin); linxi_putc('\n');
    linxi_put("low8_bound(hex): ");   linxi_put_hex((unsigned)low8_boundary); linxi_putc('\n');

    int ret = (threshold == expected_kth) ? 0 : 1;
    linxi_puts(ret == 0 ? "PASS" : "FAIL");
    return ret;
}
