#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "solution/normalization/rms_norm/rms_norm_dynamic_m_R_simt.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef PE_NUM
#define PE_NUM 1
#endif
#ifndef G_A
#define G_A 128
#endif
#ifndef G_R
#define G_R 8192
#endif

namespace {
struct SimtTilingData { int64_t g_a; int64_t g_r; int64_t powR; int64_t tile_a; int64_t tile_r; };
constexpr int64_t rms_tile_a(int64_t global_a, int64_t pe_num) {
    return global_a > 0 && pe_num > 0 ? 1 : 0;
}
constexpr int64_t rms_pow_r(int64_t reduce_size) {
    int64_t p = 1;
    while (p <= (reduce_size - 1) / 2) p <<= 1;
    return p;
}
constexpr int64_t rms_tile_r(int64_t reduce_size) {
    // tile_r denotes the total number of elements in one linear R block. The
    // kernel maps 512 elements to the physical [32,16] Tile.
    constexpr int64_t kMaxTileR = 512;
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
    constexpr int64_t kPowR = rms_pow_r(G_R);
    constexpr int64_t kTileR = rms_tile_r(G_R);
    static_assert(G_A > 0 && G_R > 0);
    static_assert(kTileA > 0 && kTileR > 0 && kTileR <= 512);
    SimtTilingData tiling_info = {
        G_A, G_R, kPowR, kTileA, kTileR};

    const int64_t g_a = tiling_info.g_a;
    const int64_t g_r = tiling_info.g_r;

    static dtype input_buf[G_A * G_R];
    static dtype gamma_buf[G_R];
    static dtype output_buf[G_A * G_R];
    dtype *input = input_buf;
    dtype *gamma = gamma_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                       static_cast<size_t>(g_a) * g_r * sizeof(dtype));
        readBinaryFile(CHK_DIR "/gamma.bin", (uint8_t *)gamma,
                       static_cast<size_t>(g_r) * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    rms_norm_dynamic_m_R_simt<dtype, PE_NUM>(input, gamma, &tiling_info, output);

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
