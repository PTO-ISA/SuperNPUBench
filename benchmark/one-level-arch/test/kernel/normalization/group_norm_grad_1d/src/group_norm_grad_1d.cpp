#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/group_norm_grad_1d/group_norm_grad_1d_pto.hpp"

#ifndef DType
#define DType __half
#endif

// Dynamic 4PE validation: HxW==1, N=512, C=64, G=8, D=8.
#ifndef N_BATCH
#define N_BATCH 512
#endif
#ifndef C_CH
#define C_CH 64
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
        (512 + static_cast<int64_t>(sizeof(dtype)) - 1) /
        static_cast<int64_t>(sizeof(dtype));
    constexpr int64_t kTileCapacity =
        kDtypeCapacity > 128 ? kDtypeCapacity : 128;
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

    // tiling: {N, C, G, tile_d}
    constexpr int64_t kTileD = group_norm_1d_tile_d<dtype>(C_CH, G_GRP);
    static_assert(N_BATCH > 0 && C_CH > 0 && G_GRP > 0);
    static_assert(C_CH % G_GRP == 0 && kTileD > 0);
    int64_t tiling_info[4] = {N_BATCH, C_CH, G_GRP, kTileD};

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

    group_norm_grad_1d<dtype, PE_NUM>(dy, x, mean, rstd, gamma, tiling_info,
                                      dx, dgamma, dbeta);

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
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
