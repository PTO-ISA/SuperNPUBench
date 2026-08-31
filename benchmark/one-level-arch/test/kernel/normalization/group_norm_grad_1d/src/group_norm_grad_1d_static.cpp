#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/group_norm_grad_1d/group_norm_grad_1d_pto.hpp"

#ifndef DType
#define DType __half
#endif

// Same as dynamic group_norm_grad_1d.cpp: N=8 C=64 G=8, tile_d=-1 → D=8
#ifndef N_BATCH
#define N_BATCH 8
#endif
#ifndef C_CH
#define C_CH 64
#endif
#ifndef G_GRP
#define G_GRP 8
#endif
#ifndef TILE_D
#define TILE_D 8
#endif
#ifndef PE_NUM
#define PE_NUM 1
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
                   static_cast<size_t>(N_BATCH) * C_CH * sizeof(dtype));
    readBinaryFile(CHK_DIR "/x.bin", (uint8_t *)x,
                   static_cast<size_t>(N_BATCH) * C_CH * sizeof(dtype));
    readBinaryFile(CHK_DIR "/mean.bin", (uint8_t *)mean,
                   static_cast<size_t>(N_BATCH) * G_GRP * sizeof(float));
    readBinaryFile(CHK_DIR "/rstd.bin", (uint8_t *)rstd,
                   static_cast<size_t>(N_BATCH) * G_GRP * sizeof(float));
    readBinaryFile(CHK_DIR "/gamma.bin", (uint8_t *)gamma,
                   static_cast<size_t>(C_CH) * sizeof(dtype));
    input_ready = 1;
  } else {
    while (input_ready == 0) {
    }
  }
#endif

  group_norm_grad_1d<dtype, N_BATCH, C_CH, G_GRP, TILE_D, PE_NUM>(
      dy, x, mean, rstd, gamma, dx, dgamma, dbeta);

#ifdef RES_CHECK
  kernel_done[tid] = 1;
  if (tid == 0) {
    for (int pe = 0; pe < PE_NUM; ++pe) {
      while (kernel_done[pe] == 0) {
      }
    }
    writeBinaryFile(CHK_DIR "/dx.bin", (uint8_t *)dx,
                    static_cast<size_t>(N_BATCH) * C_CH * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dgamma.bin", (uint8_t *)dgamma,
                    static_cast<size_t>(C_CH) * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dbeta.bin", (uint8_t *)dbeta,
                    static_cast<size_t>(C_CH) * sizeof(dtype));
    output_written = 1;
  } else {
    while (output_written == 0) {
    }
  }
#endif
}
