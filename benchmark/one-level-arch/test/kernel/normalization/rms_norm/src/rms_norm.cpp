#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/rms_norm/rms_norm_pto.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

#ifndef PE_NUM
#define PE_NUM 1
#endif
#ifndef G_A
#define G_A 512
#endif
#ifndef G_R
#define G_R 8192
#endif

namespace {
constexpr int64_t rms_tile_a(int64_t global_a, int64_t pe_num) {
    return global_a >= pe_num ? 1 : 0;
}
constexpr int64_t rms_tile_r(int64_t reduce_size) {
    constexpr int64_t kMaxTileR = 8192;
    return reduce_size < kMaxTileR ? reduce_size : kMaxTileR;
}
} // namespace

#ifdef RES_CHECK
namespace {
volatile uint32_t input_ready = 0;
volatile uint32_t kernel_done[PE_NUM] = {};
volatile uint32_t output_written = 0;
} // namespace
#endif

int main() {
    using dtype = DType;

    // tiling_info is always the host-visible full shape. PE partitioning is
    // entirely owned by the kernel.
    constexpr int64_t kTileA = rms_tile_a(G_A, PE_NUM);
    constexpr int64_t kTileR = rms_tile_r(G_R);
    static_assert(G_A > 0 && G_R > 0 && G_A % PE_NUM == 0);
    static_assert(kTileA > 0 && kTileR == G_R);
    int64_t tiling_info[4] = {G_A, G_R, kTileA, kTileR};

    const int64_t g_a = tiling_info[0];
    const int64_t g_r = tiling_info[1];

    static dtype input_buf[G_A * G_R];
    static dtype output_buf[G_A * G_R];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                       static_cast<size_t>(g_a) * g_r * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    rms_norm<dtype, PE_NUM>(input, tiling_info, output, EPS);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                        static_cast<size_t>(g_a) * g_r * sizeof(dtype));
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
