#include <cstdint>

#include "benchmark.h"
#include "multi_thread/topk/topk_tiled.hpp"
#include "topk_tiled_driver.h"

namespace {
topk_tiled_test::Buffers buffers{};
MultiThreadResCheckSync res_check_sync{};
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    topk_tiled_test::prepare(buffers, res_check_sync, tid);

    BENCHSTART;
    topk_tiled::run(buffers.output, buffers.errors, buffers.input,
                    buffers.starts, buffers.ends, buffers.scratch,
                    &buffers.tiling);
    BENCHEND;

    return topk_tiled_test::finish(buffers, res_check_sync, tid);
}
