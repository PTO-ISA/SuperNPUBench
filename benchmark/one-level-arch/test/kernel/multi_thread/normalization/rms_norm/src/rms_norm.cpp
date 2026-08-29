#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "multi_thread/normalization/rms_norm/rms_norm.hpp"

namespace {
constexpr int kPeCount = 4;
constexpr int kRowsPerPe = 1;
constexpr int kColumns = 8192;
alignas(4096) __half input[kPeCount * kRowsPerPe * kColumns];
alignas(4096) __half output[kPeCount * kRowsPerPe * kColumns];
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
    rms_norm<__half, kRowsPerPe, kPeCount * kRowsPerPe,
             kColumns, 1, kColumns>(input, output);
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
