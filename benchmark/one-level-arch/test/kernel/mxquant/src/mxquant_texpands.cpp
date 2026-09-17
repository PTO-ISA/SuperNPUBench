#include "basic_op/mxquant/mxquant_texpands.hpp"

#include "mxquant_driver.h"

namespace {

// TEXPANDS-only build: used to subtract the input-materialisation cost from
// mxquant_compute (see mxquant_texpands.hpp).  Timing-only.
constexpr int kTotalRows = mxquant_texpands::kTotalRows;
constexpr int kCols = mxquant_texpands::kCols;
constexpr int kBlocksPerRow = mxquant_texpands::kBlocksPerRow;

mxquant_test::Buffers<kTotalRows, kCols, kBlocksPerRow> buffers{};
MultiThreadResCheckSync res_check_sync{};

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    mxquant_test::prepare(buffers, res_check_sync, tid, mxquant_texpands::kBlock);

    BENCHSTART;
    mxquant_texpands::run(buffers.output, buffers.scales, buffers.input);
    BENCHEND;

    return mxquant_test::finish(buffers, res_check_sync, tid);
}
