#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/group_norm_grad/group_norm_grad_pto.hpp"

#ifndef DType
#define DType __half
#endif

// Static coverage: N=2 C=16 G=4 HxW=16 tile_hw=8
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

  constexpr int64_t kElems = static_cast<int64_t>(N_BATCH) * C_CH * HxW_SZ;
  constexpr int64_t kWs = 2 * static_cast<int64_t>(N_BATCH) * C_CH +
                          2 * static_cast<int64_t>(N_BATCH) * G_GRP;

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
  const uint32_t tid = get_thread_idx();
  if (tid == 0) {
    readBinaryFile(CHK_DIR "/dy.bin", (uint8_t *)dy,
                   static_cast<size_t>(kElems) * sizeof(dtype));
    readBinaryFile(CHK_DIR "/x.bin", (uint8_t *)x,
                   static_cast<size_t>(kElems) * sizeof(dtype));
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

  group_norm_grad<dtype, N_BATCH, C_CH, G_GRP, HxW_SZ, TILE_HW, PE_NUM>(
      dy, x, mean, rstd, gamma, dx, dgamma, dbeta, workspace);

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
