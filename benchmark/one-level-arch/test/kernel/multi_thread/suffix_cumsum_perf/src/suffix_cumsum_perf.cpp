// Micro-benchmark that isolates the histogram_cumsum_m32 operator so it can be
// measured on the cycle model (gfsim) without the rest of the topk kernel.
//
// A single per-PE hist[384] is initialized once, then
// histogram_cumsum_m32::suffix_cumsum is called kIters times between
// BENCHSTART/BENCHEND.  The kernel only measures the cumsum block stream.
//
// hist[0:256] is the histogram; hist[256:384] is the zero pad the shifted
// TLOAD windows read past the sentinel.  The values grow across iterations
// (each call re-suffix-sums the previous result); the block stream and hence
// the timing are value-independent.

#include <cstdint>

#include "benchmark.h"
#include "multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp"

namespace {

#ifndef CUMSUM_ITERS
#define CUMSUM_ITERS 32
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

    int32_t *hist = g_buf.hist;
    for (int i = 0; i < 256; ++i) hist[i] = (i % 7) + 1;
    for (int i = 256; i < 384; ++i) hist[i] = 0;

    volatile int iters = kIters;

    BENCHSTART;
    for (int k = 0; k < iters; ++k) {
        histogram_cumsum_m32::suffix_cumsum(hist);
    }
    BENCHEND;

    return 0;
}
