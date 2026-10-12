// 256-bin suffix cumsum operator micro-test.
//
// One host-written hist[384] per PE (256 bins + 128-word zero pad); each PE
// runs histogram_cumsum_m32::suffix_cumsum once, in place. The res_check build
// exports the result for run_histogram_cumsum_m32_check.py, which compares it
// to the scalar suffix sum.

#include <cstdint>

#include "benchmark.h"
#include "histogram_cumsum_m32_driver.h"
#include "multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.hpp"

namespace {
histogram_cumsum_m32_test::Buffers buffers{};
MultiThreadResCheckSync res_check_sync{};
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    histogram_cumsum_m32_test::prepare(buffers, res_check_sync, tid);

    if (tid < histogram_cumsum_m32_test::kRows) {
        BENCHSTART;
        histogram_cumsum_m32::suffix_cumsum(buffers.hist[tid]);
        BENCHEND;
    }

    return histogram_cumsum_m32_test::finish(buffers, res_check_sync, tid);
}
