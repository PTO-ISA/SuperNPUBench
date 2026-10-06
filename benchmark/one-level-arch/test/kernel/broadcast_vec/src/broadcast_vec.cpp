#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/broadcast/broadcast_vec.hpp"

namespace {
#ifdef BROADCAST_VEC_2D_1334_129
alignas(4096) __half input[1334];
alignas(4096) __half output[1334 * 129];
#else
alignas(4096) __half input[8192 * 16];
alignas(4096) __half output[8192 * 8 * 16];
#endif
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
#ifdef BROADCAST_VEC_2D_1334_129
    supernpu::multi_thread::broadcast_vec_2d<__half, 1334, 129, 16>(
        input, output);
#else
    supernpu::multi_thread::broadcast_vec_3d<__half, 8192, 8, 16, 16>(
        input, output);
#endif
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
