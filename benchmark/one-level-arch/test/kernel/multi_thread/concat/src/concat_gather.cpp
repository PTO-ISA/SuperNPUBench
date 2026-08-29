#include <cstddef>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "multi_thread/concat/concat_gather.hpp"

namespace {
alignas(4096) float input[512];
alignas(4096) float output[512];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    std::size_t input_shape[2] = {16, 16};
    std::size_t output_shape[2] = {16, 32};
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::concat_gather<float, 8, 512, 512, 128, 2, 1>(
        input, output, input_shape, output_shape);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }
#endif
    return 0;
}
