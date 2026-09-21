#include <cstdint>

#include "benchmark.h"
#include "multi_thread/topk/topk.hpp"
#include "topk_driver.h"

namespace {
topk_test::Buffers buffers{};
MultiThreadResCheckSync res_check_sync{};
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    topk_test::prepare(buffers, res_check_sync, tid);

    BENCHSTART;
    topk_radix::run(buffers.output, buffers.errors, buffers.input,
                    buffers.starts, buffers.ends, buffers.scratch);
    BENCHEND;

    return topk_test::finish(buffers, res_check_sync, tid);
}
