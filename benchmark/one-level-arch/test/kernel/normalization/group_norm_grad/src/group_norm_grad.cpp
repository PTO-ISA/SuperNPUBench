#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/group_norm_grad/group_norm_grad_pto.hpp"

#ifndef DType
#define DType __half
#endif

// Dynamic 4PE validation: N=32, C=16, G=8, HxW=512.
#ifndef N_BATCH
#define N_BATCH 32
#endif
#ifndef C_CH
#define C_CH 16
#endif
#ifndef G_GRP
#define G_GRP 8
#endif
#ifndef HxW_SZ
#define HxW_SZ 512
#endif
#ifndef PE_NUM
#define PE_NUM 1
#endif

namespace {
template <typename dtype>
constexpr int64_t group_norm_tile_hw(int64_t spatial_size) {
    constexpr int64_t kTileCapacity = 512;
    return spatial_size < kTileCapacity ? spatial_size : kTileCapacity;
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

    // tiling: {N, C, G, HxW, tile_hw}
    constexpr int64_t kTileHw = group_norm_tile_hw<dtype>(HxW_SZ);
    static_assert(N_BATCH > 0 && C_CH > 0 && G_GRP > 0 && HxW_SZ > 0);
    static_assert(C_CH % G_GRP == 0 && kTileHw > 0);
    int64_t tiling_info[5] = {N_BATCH, C_CH, G_GRP, HxW_SZ, kTileHw};

    const int64_t N = tiling_info[0];
    const int64_t C = tiling_info[1];
    const int64_t G = tiling_info[2];
    const int64_t HxW = tiling_info[3];

    constexpr int64_t kElems = N_BATCH * C_CH * HxW_SZ;
    constexpr int64_t kWs =
        2 * N_BATCH * C_CH + 2 * N_BATCH * G_GRP;

    static dtype dy_buf[kElems];
    static dtype x_buf[kElems];
    static float mean_buf[N_BATCH * G_GRP];
    static float rstd_buf[N_BATCH * G_GRP];
    static dtype gamma_buf[C_CH];
    static dtype dx_buf[kElems];
    static dtype dgamma_buf[C_CH];
    static dtype dbeta_buf[C_CH];
    static float workspace_buf[kWs];

    dtype *dy = dy_buf;
    dtype *x = x_buf;
    float *mean = mean_buf;
    float *rstd = rstd_buf;
    dtype *gamma = gamma_buf;
    dtype *dx = dx_buf;
    dtype *dgamma = dgamma_buf;
    dtype *dbeta = dbeta_buf;
    float *workspace = workspace_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = gn_grad::read_pe_id();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/dy.bin", (uint8_t *)dy,
                       static_cast<size_t>(kElems) * sizeof(dtype));
        readBinaryFile(CHK_DIR "/x.bin", (uint8_t *)x,
                       static_cast<size_t>(kElems) * sizeof(dtype));
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

    group_norm_grad<dtype, PE_NUM>(dy, x, mean, rstd, gamma, tiling_info, dx,
                                  dgamma, dbeta, workspace);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/dx.bin", (uint8_t *)dx,
                        static_cast<size_t>(kElems) * sizeof(dtype));
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
