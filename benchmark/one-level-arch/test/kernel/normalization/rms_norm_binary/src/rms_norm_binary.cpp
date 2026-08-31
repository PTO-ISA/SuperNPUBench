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

// Spec: default shape [g_a, g_r] = [1, 8192]; 4PE validation uses [4, 8192], fp16
// tiling: tile_a=1, tile_r=1024, pow_r=4096
//   pow_r is 2^n and pow_r < g_r <= 2*pow_r
#ifndef G_A
#define G_A 1
#endif
#ifndef G_R
#define G_R 8192
#endif
#ifndef TILE_A
#define TILE_A 1
#endif
#ifndef TILE_R
#define TILE_R 1024
#endif
#ifndef POW_R
#define POW_R 4096
#endif
// Must match rms_bin::kWsCols / kMaxLevels
#ifndef K_WS_COLS
#define K_WS_COLS 128
#endif
#ifndef K_MAX_LEVELS
#define K_MAX_LEVELS 6
#endif

#ifdef RES_CHECK
namespace {
volatile uint32_t input_ready = 0;
volatile uint32_t kernel_done[PE_NUM] = {};
volatile uint32_t output_written = 0;
} // namespace
#endif

int main() {
    using dtype = DType;

    int64_t tiling_info[5] = {G_A, G_R, TILE_A, TILE_R, POW_R};

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
