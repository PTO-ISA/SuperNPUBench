#include "multi_thread/mxquant/mxquant_1024_streaming.hpp"

#include "mxquant_driver.h"

namespace {

constexpr int kTotalRows = mxquant_streaming::kTotalRows;
constexpr int kCols = mxquant_streaming::kCols;
constexpr int kBlocksPerRow = mxquant_streaming::kBlocksPerRow;

// Shared across the four PEs: the kernel slices this tensor by M internally.
mxquant_test::Buffers<kTotalRows, kCols, kBlocksPerRow> buffers{};
MultiThreadResCheckSync res_check_sync{};

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    mxquant_test::prepare(buffers, res_check_sync, tid,
                          mxquant_streaming::kBlock);

    BENCHSTART;
    mxquant_streaming::run(buffers.output, buffers.scales, buffers.input);
    BENCHEND;

    return mxquant_test::finish(buffers, res_check_sync, tid);
}
