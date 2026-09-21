// Micro-benchmark that isolates the histogram_cumsum_m32 operator so it can be
// measured on the cycle model (gfsim) without the rest of the topk kernel.
//
// A single per-PE hist[384] is initialized once, then
// histogram_cumsum_m32::suffix_cumsum runs between BENCHSTART/BENCHEND.  One
// call already computes the complete 256-bin cumsum, so the default is a
// single pass; CUMSUM_ITERS can repeat it to lengthen the trace (the values
// then grow and the block stream stays value-independent).
//
// hist[0:256] is the histogram; hist[256:384] is the zero pad the grouped
// TLOAD reads past the sentinel.

#include <cstdint>

#include "benchmark.h"
#include "multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp"

namespace {

#ifndef CUMSUM_ITERS
#define CUMSUM_ITERS 1
#endif
constexpr int kIters = CUMSUM_ITERS;

struct alignas(4096) Buffer {
    int32_t hist[384];
};

Buffer g_buf;

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    if (tid != 0) return 0;

    // g_buf is zero-initialized .bss; the operator's block stream is
    // value-independent, so no input setup is needed and gfsim's whole-run
    // Total Cycles is the operator cost.
    int32_t *hist = g_buf.hist;

    BENCHSTART;
#if CUMSUM_ITERS > 1
    volatile int iters = kIters;
    for (int k = 0; k < iters; ++k) {
        histogram_cumsum_m32::suffix_cumsum(hist);
    }
#else
    histogram_cumsum_m32::suffix_cumsum(hist);
#endif
    BENCHEND;

    return 0;
}
