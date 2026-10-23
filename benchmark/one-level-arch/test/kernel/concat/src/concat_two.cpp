#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/concat/concat_two.hpp"

namespace {
using Data = DTYPE;
constexpr std::size_t kLeft = CONCAT_OUTER * CONCAT_LEFT * CONCAT_INNER;
constexpr std::size_t kRight = CONCAT_OUTER * CONCAT_RIGHT * CONCAT_INNER;
constexpr std::size_t kOutput = kLeft + kRight;
constexpr std::size_t kGuard = 32;
alignas(4096) Data left[kLeft ? kLeft : 1];
alignas(4096) Data right[kRight ? kRight : 1];
alignas(4096) Data guarded_output[kOutput + 2 * kGuard];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
volatile std::uint32_t inputs_ok = 0;
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        inputs_ok = readBinaryFile(CHK_DIR "/left.bin",
                                   reinterpret_cast<uint8_t *>(left), kLeft * sizeof(Data)) &&
                    readBinaryFile(CHK_DIR "/right.bin",
                                   reinterpret_cast<uint8_t *>(right), kRight * sizeof(Data)) &&
                    readBinaryFile(CHK_DIR "/output.bin",
                                   reinterpret_cast<uint8_t *>(guarded_output), sizeof(guarded_output));
    }
    res_check_publish_inputs(res_check_sync, tid);
    if (!inputs_ok) {
        return 1;
    }
#endif
    BENCHSTART;
    supernpu::multi_thread::concat_two<Data, CONCAT_OUTER, CONCAT_LEFT,
        CONCAT_RIGHT, CONCAT_INNER, CONCAT_TILE>(left, right, guarded_output + kGuard);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0 && !writeBinaryFile(CHK_DIR "/output.bin",
            reinterpret_cast<uint8_t *>(guarded_output), sizeof(guarded_output))) {
        return 1;
    }
#endif
    return 0;
}
