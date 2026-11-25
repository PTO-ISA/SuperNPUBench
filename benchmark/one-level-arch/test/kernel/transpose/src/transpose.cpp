#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/transpose/transpose_u16_shapes.hpp"

namespace {
alignas(4096) std::uint16_t input[64 * 64];
alignas(4096) std::uint16_t output[64 * 64];
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
    supernpu::multi_thread::TransposeU16Transfer<32, 1, 64, false> transfer;
    for (unsigned col = tid; col < 64; col += 4) {
        transfer.copy(input + col, output + col * 64);
        transfer.copy(input + 32 * 64 + col, output + col * 64 + 32);
    }
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
