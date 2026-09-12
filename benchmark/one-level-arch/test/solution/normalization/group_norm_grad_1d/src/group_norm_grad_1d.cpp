#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "solution/normalization/group_norm_grad_1d/group_norm_grad_1d.hpp"

#ifndef DType
#define DType __half
#endif

// Dynamic 4PE validation: HxW==1, N=256, C=256, G=8, D=32.
#ifndef N_BATCH
#define N_BATCH 256
#endif
#ifndef C_CH
#define C_CH 256
#endif
#ifndef G_GRP
#define G_GRP 8
#endif
#ifndef PE_NUM
#define PE_NUM 1
#endif

namespace {
template <typename dtype>
constexpr int64_t group_norm_1d_tile_d(int64_t channels, int64_t groups) {
    constexpr int64_t kDtypeCapacity =
        (32768 + static_cast<int64_t>(sizeof(dtype)) - 1) /
        static_cast<int64_t>(sizeof(dtype));
    constexpr int64_t kTileCapacity =
        kDtypeCapacity < 8192 ? kDtypeCapacity : 8192;
    const int64_t group_width = channels / groups;
    return group_width < kTileCapacity ? group_width : kTileCapacity;
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

    // tiling: {N, C, G, tile_d, tile_g, gb_tile_d, gb_tile_g}
    constexpr int64_t kTileD = group_norm_1d_tile_d<dtype>(C_CH, G_GRP);
    static_assert(N_BATCH > 0 && C_CH > 0 && G_GRP > 0);
    static_assert(C_CH % G_GRP == 0 && kTileD > 0);
    // Small-D physical row stride is 256 FP32 elements: 32 KiB / 1024 = 32 rows.
    constexpr int64_t kTileG = C_CH / G_GRP <= 256 ? (G_GRP < 32 ? G_GRP : 32) : 1;
    // Separate Stage B tiling, with optional validation overrides.
#ifndef GB_TILE_D
#define GB_TILE_D 0
#endif
#ifndef GB_TILE_G
#define GB_TILE_G 0
#endif
    constexpr int64_t kGbTileD = GB_TILE_D > 0 ? GB_TILE_D : kTileD;
    constexpr int64_t kGbRowCapacity = 32768 / (256 * sizeof(float));
    constexpr int64_t kGbTileG = GB_TILE_G > 0 ? GB_TILE_G :
        (kGbTileD <= 256 ? (G_GRP < kGbRowCapacity ? G_GRP : kGbRowCapacity) : 1);
    static_assert(kGbTileD > 0 && kGbTileD <= 8192);
    static_assert(kGbTileG > 0 && kGbTileG <= kGbRowCapacity);
    static_assert(kGbTileD <= 256 || kGbTileG == 1);
    int64_t tiling_info[7] = {N_BATCH, C_CH, G_GRP, kTileD, kTileG,
                              kGbTileD, kGbTileG};

    const int64_t N = tiling_info[0];
    const int64_t C = tiling_info[1];
    const int64_t G = tiling_info[2];

    static dtype dy_buf[N_BATCH * C_CH];
    static dtype x_buf[N_BATCH * C_CH];
    static float mean_buf[N_BATCH * G_GRP];
    static float rstd_buf[N_BATCH * G_GRP];
    static dtype gamma_buf[C_CH];
    static dtype dx_buf[N_BATCH * C_CH];
    static dtype dgamma_buf[C_CH];
    static dtype dbeta_buf[C_CH];
    static float params_workspace[2 * N_BATCH * G_GRP];

    dtype *dy = dy_buf;
    dtype *x = x_buf;
    float *mean = mean_buf;
    float *rstd = rstd_buf;
    dtype *gamma = gamma_buf;
    dtype *dx = dx_buf;
    dtype *dgamma = dgamma_buf;
    dtype *dbeta = dbeta_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/dy.bin", (uint8_t *)dy,
                       static_cast<size_t>(N) * C * sizeof(dtype));
        readBinaryFile(CHK_DIR "/x.bin", (uint8_t *)x,
                       static_cast<size_t>(N) * C * sizeof(dtype));
        readBinaryFile(CHK_DIR "/mean.bin", (uint8_t *)mean,
                       static_cast<size_t>(N) * G * sizeof(float));
        readBinaryFile(CHK_DIR "/rstd.bin", (uint8_t *)rstd,
                       static_cast<size_t>(N) * G * sizeof(float));
        readBinaryFile(CHK_DIR "/gamma.bin", (uint8_t *)gamma,
                       static_cast<size_t>(C) * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    group_norm_grad_1d_fused_params<dtype, PE_NUM>(
        dy, x, mean, rstd, gamma, tiling_info, params_workspace);
    // Each PE consumes only the parameter groups it produced above.
    // Add a stage barrier if parameters and dx use different PE ownership.
    group_norm_grad_1d_dx<dtype, PE_NUM>(
        dy, x, rstd, gamma, tiling_info, params_workspace, dx);
    group_norm_grad_1d_gamma_beta<dtype, PE_NUM>(
        dy, x, mean, rstd, tiling_info, dgamma, dbeta);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/dx.bin", (uint8_t *)dx,
                        static_cast<size_t>(N) * C * sizeof(dtype));
        writeBinaryFile(CHK_DIR "/dgamma.bin", (uint8_t *)dgamma,
                        static_cast<size_t>(C) * sizeof(dtype));
        writeBinaryFile(CHK_DIR "/dbeta.bin", (uint8_t *)dbeta,
                        static_cast<size_t>(C) * sizeof(dtype));
        writeBinaryFile(CHK_DIR "/fused_params.bin", (uint8_t *)params_workspace,
                        static_cast<size_t>(2 * N * G) * sizeof(float));
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
