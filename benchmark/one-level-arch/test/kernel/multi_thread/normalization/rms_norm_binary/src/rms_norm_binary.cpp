#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "multi_thread/normalization/rms_norm_binary/rms_norm_binary.hpp"

namespace {
alignas(4096) __half input[4 * 8192];
alignas(4096) __half output[4 * 8192];
alignas(4096) float
    workspace[4 * rms_bin::kMaxLevels * rms_bin::kWsCols];
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
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::rms_norm_binary<
        __half, 4, 8192, 1, 1024, 4096>(input, output, workspace);
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
