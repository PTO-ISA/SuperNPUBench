#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/rms_norm_binary/rms_norm_binary_pto.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

#ifndef PE_NUM
#define PE_NUM 1
#endif

// Dynamic 4PE validation shape: [16, 16384], fp16.
#ifndef G_A
#define G_A 16
#endif
#ifndef G_R
#define G_R 16384
#endif
// Must match rms_bin::kWsCols / kMaxLevels
#ifndef K_WS_COLS
#define K_WS_COLS 1
#endif
#ifndef K_MAX_LEVELS
#define K_MAX_LEVELS 6
#endif

namespace {
constexpr int64_t floor_power_of_two(int64_t value) {
    int64_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}
constexpr int64_t binary_tile_r(int64_t reduce_size) {
    constexpr int64_t kMaxTileR = 1024;
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

    constexpr int64_t kTileA = 1;
    constexpr int64_t kTileR = binary_tile_r(G_R);
    constexpr int64_t kPowR = floor_power_of_two(G_R - 1);
    static_assert(G_A > 0 && G_R > 1 && G_A % PE_NUM == 0);
    static_assert(kPowR < G_R && G_R <= 2 * kPowR);
    int64_t tiling_info[5] = {G_A, G_R, kTileA, kTileR, kPowR};

    const int64_t g_a = tiling_info[0];
    const int64_t g_r = tiling_info[1];

    static dtype input_buf[G_A * G_R];
    static dtype output_buf[G_A * G_R];
    static float workspace_buf[K_MAX_LEVELS * G_A * K_WS_COLS];
    dtype *input = input_buf;
    dtype *output = output_buf;
    float *workspace = workspace_buf;

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

    rms_norm_binary<dtype>(input, tiling_info, output, workspace, EPS);

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
