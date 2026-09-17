#include "basic_op/mxquant/mxquant_compute.hpp"

#include "mxquant_driver.h"

namespace {

// Compute-only with the ideal |x| = TABS.  TIMING-ONLY: gfrun's TABS CUBE_M32
// writeback is permuted (issue #678), so this variant is never res_checked.
constexpr int kTotalRows = mxquant_compute::kTotalRows;
constexpr int kCols = mxquant_compute::kCols;
constexpr int kBlocksPerRow = mxquant_compute::kBlocksPerRow;

mxquant_test::Buffers<kTotalRows, kCols, kBlocksPerRow> buffers{};
MultiThreadResCheckSync res_check_sync{};

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    mxquant_test::prepare(buffers, res_check_sync, tid, mxquant_compute::kBlock);

    BENCHSTART;
    mxquant_compute::run<true, true>(buffers.output, buffers.scales,
                                     buffers.input);
    BENCHEND;

    return mxquant_test::finish(buffers, res_check_sync, tid);
}
