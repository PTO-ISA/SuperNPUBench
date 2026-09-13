#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "solution/normalization/group_norm_grad/group_norm_grad_dynamic.hpp"

#ifndef DType
#define DType __half
#endif

// Dynamic 4PE validation: N=2, C=32, G=8, HxW=2024.
#ifndef N_BATCH
#define N_BATCH 2
#endif
#ifndef C_CH
#define C_CH 32
#endif
#ifndef G_GRP
#define G_GRP 8
#endif
#ifndef HxW_SZ
#define HxW_SZ 2024
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

  static_assert(N_BATCH > 0 && C_CH > 0 && G_GRP > 0 && HxW_SZ > 0 &&
                C_CH % G_GRP == 0);
  constexpr int64_t D = C_CH / G_GRP;
  constexpr int64_t rh = HxW_SZ < 512 ? HxW_SZ : 512, rc = 1;
  constexpr int64_t dh = HxW_SZ < 8192 ? HxW_SZ : 8192;
  constexpr int64_t dc = dh <= 256 ? (D < 32 ? D : 32) : 1;
  constexpr int64_t bd = D < 8192 ? D : 8192,
                    bg = bd <= 256 ? (G_GRP < 32 ? G_GRP : 32) : 1;
  int64_t tiling_info[10] = {N_BATCH, C_CH, G_GRP, HxW_SZ, rh,
                             rc,      dh,   dc,    bd,     bg};

  const int64_t N = tiling_info[0];
  const int64_t C = tiling_info[1];
  const int64_t G = tiling_info[2];

  constexpr int64_t kElems = N_BATCH * C_CH * HxW_SZ;
  constexpr int64_t kWs = gn_grad::workspace_elems(N_BATCH, C_CH, G_GRP);

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

  group_norm_grad_spatial<dtype, PE_NUM>(dy, x, tiling_info, workspace);
  group_norm_grad_fused_params<dtype, PE_NUM>(gamma, mean, rstd, tiling_info,
                                              workspace);
  group_norm_grad_dx<dtype, PE_NUM>(dy, x, gamma, rstd, tiling_info, workspace,
                                    dx);
  group_norm_grad_gamma_beta<dtype, PE_NUM>(mean, rstd, tiling_info, workspace,
                                            dgamma, dbeta);

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
