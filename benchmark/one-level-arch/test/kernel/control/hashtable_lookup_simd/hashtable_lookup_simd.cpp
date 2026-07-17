#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "template_asm.h"
#include "linx_print.h"

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

    // Verify results. Output goes through linxi_put (raw UART MMIO), which is
    // visible on both qemu virt and gfsim ("linx_uart:" lines) without any
    // libc/syscall dependency -- so this runs identically under both back-ends.
    int match = 0;
    int mismatch_count = 0;
    for (int i = 0; i < kNum; i++) {
        if (g_output[i] == g_lookup_values[i]) {
            match++;
        } else {
            mismatch_count++;
        }
    }

    // NOTE: counts are printed in hex via linxi_put_hex (shift/mask based). The
    // decimal helper linxi_put_i64 uses scalar unsigned divide (divu), which does
    // not terminate on the current model, so it is avoided here.
    linxi_puts("=== hashtable_lookup_simd ===");
    linxi_put("Match(hex): ");
    linxi_put_hex((unsigned long long)(unsigned)match);
    linxi_put(" / ");
    linxi_put_hex((unsigned long long)(unsigned)kNum);
    linxi_putc('\n');

    int ret = (match == kNum) ? 0 : 1;
    linxi_puts(ret == 0 ? "PASS" : "FAIL");
    return ret;
}
