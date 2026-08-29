#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "multi_thread/conv2d/v300_conv2d.hpp"

namespace {
alignas(4096) float input[16 * 8 * 8];
alignas(4096) float weight[16 * 16];
alignas(4096) float output[16 * 8 * 8];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
        readBinaryFile(CHK_DIR "/weight.bin", reinterpret_cast<uint8_t *>(weight),
                       sizeof(weight));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::conv2d::conv2d_1x1<
        float, 16, 8, 8, 16, 16, 16, 16>(output, input, weight);
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
