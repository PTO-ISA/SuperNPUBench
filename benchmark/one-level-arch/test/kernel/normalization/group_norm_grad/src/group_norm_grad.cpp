#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/group_norm_grad/group_norm_grad_pto.hpp"

#ifndef DType
#define DType __half
#endif

// Default: HxW>1, N=2, C=16, G=4 → D=4, HxW=16
#ifndef N_BATCH
#define N_BATCH 2
#endif
#ifndef C_CH
#define C_CH 16
#endif
#ifndef G_GRP
#define G_GRP 4
#endif
#ifndef HxW_SZ
#define HxW_SZ 16
#endif
#ifndef TILE_HW
#define TILE_HW 8
#endif

int main() {
    using dtype = DType;

    // tiling: {N, C, G, HxW, tile_hw}
    int64_t tiling_info[5] = {N_BATCH, C_CH, G_GRP, HxW_SZ, TILE_HW};

    const int64_t N = tiling_info[0];
    const int64_t C = tiling_info[1];
    const int64_t G = tiling_info[2];
    const int64_t HxW = tiling_info[3];

    constexpr int64_t kElems = N_BATCH * C_CH * HxW_SZ;
    constexpr int64_t kWs =
        2 * N_BATCH * C_CH + 2 * N_BATCH * G_GRP; // ds+db+c2+c3

    dtype dy_buf[kElems];
    dtype x_buf[kElems];
    float mean_buf[N_BATCH * G_GRP];
    float rstd_buf[N_BATCH * G_GRP];
    dtype gamma_buf[C_CH];
    dtype dx_buf[kElems];
    dtype dgamma_buf[C_CH];
    dtype dbeta_buf[C_CH];
    float workspace_buf[kWs];

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
#endif

    group_norm_grad<dtype>(dy, x, mean, rstd, gamma, tiling_info, dx, dgamma,
                           dbeta, workspace);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/dx.bin", (uint8_t *)dx,
                    static_cast<size_t>(kElems) * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dgamma.bin", (uint8_t *)dgamma,
                    static_cast<size_t>(C) * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dbeta.bin", (uint8_t *)dbeta,
                    static_cast<size_t>(C) * sizeof(dtype));
#endif
}
