// =============================================================================
// group_norm_grad.hpp — GroupNorm backward, HxW > 1 (one-level PTO)
// =============================================================================
//
// Matches PyTorch GroupNormBackwardKernelImplInternal
// (aten/src/ATen/native/cuda/group_norm_kernel.cu):
//   1) Spatial reduce  ds/db = Σ_hw (dY*X), Σ_hw dY     per (n,c)
//   2) Fused c2/c3     from ds/db over channels → workspace
//   3) dX              = (rstd*gamma)*dY + c2*X + c3
//   4) dgamma / dbeta  from ds/db over N
//
// Layout: X/dY/dX [N,C,HxW]; mean/rstd [N,G] fp32; gamma/dgamma/dbeta [C]
// workspace float[2*N*C+2*N*G]
// All pointers required (dy,x,mean,rstd,gamma,dx,dgamma,dbeta,workspace).
// tiling[10]: N,C,G,HxW,reduce_hw,reduce_c,dx_hw,dx_c,gb_d,gb_g.
// Every Tile is at most 32 KiB (FP32 CubeM32 [32,256]). Stage 1 (R=HxW) and
// stage 2 (R=D) reduce in one pass (group_norm_grad_reduce.hpp) when R is a
// power of two <= 256 or folds to [<=32, 2^k<=256]; otherwise [1,256] strips
// spill partial sums through GM. Stage 3 prefers the same HxW fold so each
// channel is one [fa,Cols] TLOAD (e.g. HxW=2048 → [8,256] = 4 KiB FP16)
// instead of HxW/256 × 512 B strips that cannot hide SoC latency.
// One kernel runs all four stages; each PE owns complete groups across N.
// Every workspace read has a producer on the same PE; no inter-PE barrier.
//
// Torch CUDA launch 总览 (NVIDIA, warp=32):
//   Step1 ComputeInternalGradientsCUDAKernel
//     grid=N*C, block=(HxW<512)?32:512
//   Step2 ComputeBackwardFusedParamsCUDAKernel
//     grid=dim3(N,G), block=(D<512)?32:512
//   Step3 dX gpu_kernel (+ optional c1)
//     block=128, vt=4(fp16)/2(fp32), grid=ceil(numel/(128*vt))
//   Step4 GammaBetaBackwardCUDAKernel1/2
//     N<=128: grid=ceil(C/256), block=256
//     N>128:  grid=ceil(C/32),  block=dim3(32,16)
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_PTO_HPP
#define SUPERNPU_GROUP_NORM_GRAD_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

#include "group_norm_grad_reduce.hpp"

namespace gn_grad {

constexpr int64_t workspace_elems(int64_t N, int64_t C, int64_t G) {
  return 2 * N * C + 2 * N * G;
}

using namespace gn_grad_reduce;

// Stage 1, HxW = Cols: channels [c, c+rows) of batch n as one [rows,HxW] Tile.
template <typename dtype, int Cols>
inline void spatial_rows(dtype *dy, dtype *x, float *ds, float *db, int64_t C,
                         int64_t n, int64_t c, int64_t rows) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using T = Tiles<dtype, Cols>;
  const int64_t off = (n * C + c) * Cols;
  GM gx(x + off, rows, Cols), gy(dy + off, rows, Cols);
  GFloat gs(ds + n * C + c, rows, 1), gb(db + n * C + c, rows, 1);
  typename T::H v(rows, Cols);
  typename T::F xf(rows, Cols), yf(rows, Cols);
  typename T::W sw(rows), bw(rows);
  TLOAD(v, gy);
  TCVT(yf, v);
  TLOAD(v, gx);
  TCVT(xf, v);
  TROWSUM(bw, yf);
  TMUL(xf, xf, yf);
  TROWSUM(sw, xf);
  TSTORE(gs, sw);
  TSTORE(gb, bw);
}

// Stage 1, HxW = fa*Cols: one channel folded to [fa,Cols].
template <typename dtype, int Cols>
inline void spatial_fold(dtype *dy, dtype *x, float *ds, float *db, int64_t C,
                         int64_t n, int64_t c, int64_t fa) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using T = Tiles<dtype, Cols>;
  const int64_t off = (n * C + c) * fa * Cols;
  GM gx(x + off, fa, Cols), gy(dy + off, fa, Cols);
  GFloat gs(ds + n * C + c, 1, 1), gb(db + n * C + c, 1, 1);
  typename T::H v(fa, Cols);
  typename T::F xf(fa, Cols), yf(fa, Cols);
  typename T::W sw(fa), bw(fa);
  typename T::S s_sum, b_sum;
  TLOAD(v, gy);
  TCVT(yf, v);
  TLOAD(v, gx);
  TCVT(xf, v);
  TROWSUM(bw, yf);
  TMUL(xf, xf, yf);
  TROWSUM(sw, xf);
  TCOLSUM(b_sum, bw);
  TCOLSUM(s_sum, sw);
  TSTORE(gs, s_sum);
  TSTORE(gb, b_sum);
}

// Stage 2, D = Cols: groups [g, g+rows) of batch n as one [rows,D] Tile.
template <typename dtype, int Cols>
inline void fused_params_rows(dtype *gamma, float *mean, float *rstd,
                              float *ds, float *db, float *c2_buf,
                              float *c3_buf, int64_t C, int64_t G, int64_t n,
                              int64_t g, int64_t rows, float s) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using T = Tiles<dtype, Cols>;
  const int64_t ng = n * G + g;
  GFloat gds(ds + n * C + g * Cols, rows, Cols);
  GFloat gdb(db + n * C + g * Cols, rows, Cols);
  GM gg(gamma + g * Cols, rows, Cols);
  GFloat gm(mean + ng, rows, 1), gr(rstd + ng, rows, 1);
  GFloat gc2(c2_buf + ng, rows, 1), gc3(c3_buf + ng, rows, 1);
  typename T::H h(rows, Cols);
  typename T::F gf(rows, Cols), sf(rows, Cols), bf(rows, Cols);
  typename T::W w1(rows), w2(rows);
  TLOAD(h, gg);
  TCVT(gf, h);
  TLOAD(sf, gds);
  TLOAD(bf, gdb);
  TMUL(sf, sf, gf);
  TROWSUM(w1, sf);
  TMUL(bf, bf, gf);
  TROWSUM(w2, bf);
  c2_c3_rows<T>(w1, w2, gm, gr, gc2, gc3, rows, s);
}

// Stage 2, D = fa*Cols: one (n,g) folded to [fa,Cols].
template <typename dtype, int Cols>
inline void fused_params_fold(dtype *gamma, float *mean, float *rstd,
                              float *ds, float *db, float *c2_buf,
                              float *c3_buf, int64_t C, int64_t G, int64_t n,
                              int64_t g, int64_t fa, float s) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using T = Tiles<dtype, Cols>;
  const int64_t ng = n * G + g, off = n * C + g * fa * Cols;
  GFloat gds(ds + off, fa, Cols), gdb(db + off, fa, Cols);
  GM gg(gamma + g * fa * Cols, fa, Cols);
  GFloat gm(mean + ng, 1, 1), gr(rstd + ng, 1, 1);
  GFloat gc2(c2_buf + ng, 1, 1), gc3(c3_buf + ng, 1, 1);
  typename T::H h(fa, Cols);
  typename T::F gf(fa, Cols), sf(fa, Cols), bf(fa, Cols);
  typename T::W w1(fa), w2(fa);
  TLOAD(h, gg);
  TCVT(gf, h);
  TLOAD(sf, gds);
  TLOAD(bf, gdb);
  TMUL(sf, sf, gf);
  TROWSUM(w1, sf);
  TMUL(bf, bf, gf);
  TROWSUM(w2, bf);
  c2_c3_fold<T>(w1, w2, gm, gr, gc2, gc3, s);
}

// ---------------------------------------------------------------------------
// Step 2: fused c2/c3 for one (n, g) → c2[ng], c3[ng]
//
// Torch: ComputeBackwardFusedParamsCUDAKernel
//   grid  = dim3(N, G)     // blockIdx.x=n, blockIdx.y=g；本函数 = 其中一个
//   block = (D < 512) ? 32 : 512
//   线程: threadIdx.x 沿 group 内通道 i∈[0,D) stride，再 block reduce
//   → c2,c3 每 (n,g) 各一个标量
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void fused_params_group(dtype *gamma, float *mean, float *rstd,
                               float *ds, float *db, float *c2_buf,
                               float *c3_buf, int64_t N, int64_t C, int64_t G,
                               int64_t D, int64_t n, int64_t g, float s) {
  const int64_t ng = n * G + g;
  const int64_t c0 = g * D;
  gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
  gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
  gm_f gc2(c2_buf + ng, static_cast<int>(N * G), 1);
  gm_f gc3(c3_buf + ng, static_cast<int>(N * G), 1);
  using scalar_tile = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1>;
  scalar_tile mean_t, rstd_t, sum1, sum2, c2, c3, loaded;
  tile_v partial(1);
  // Each (n,g) owns this output slot; overwrite with final c2 after reduction.
  gm_f partial_gm(c2_buf + ng, 1, 1);
  TLOAD(mean_t, gmean);
  TLOAD(rstd_t, grstd);
  TEXPANDS(sum1, 0.0f);
  TEXPANDS(sum2, 0.0f);
  for (int64_t d0 = 0; d0 < D; d0 += tile_f::Cols) {
    const size_t vd = D - d0 < tile_f::Cols ? D - d0 : tile_f::Cols;
    gm_f gds(ds + n * C + c0 + d0, 1, static_cast<int>(C));
    gm_f gdb(db + n * C + c0 + d0, 1, static_cast<int>(C));
    gm_h gg(gamma + c0 + d0, 1, static_cast<int>(C));
    tile_f ds_f(1, vd), db_f(1, vd), gamma_f(1, vd), t0(1, vd);
    tile_h h0(1, vd);
    TLOAD(ds_f, gds);
    TLOAD(db_f, gdb);
    TLOAD(h0, gg);
    TCVT(gamma_f, h0);
    TMUL(t0, ds_f, gamma_f);
    TROWSUM(partial, t0);
    TSTORE(partial_gm, partial);
    TLOAD(loaded, partial_gm);
    TADD(sum1, sum1, loaded);
    TMUL(t0, db_f, gamma_f);
    TROWSUM(partial, t0);
    TSTORE(partial_gm, partial);
    TLOAD(loaded, partial_gm);
    TADD(sum2, sum2, loaded);
  }

  // c2/c3 由 block 内 thread 0（归约后）写出；此处标量 tile 完成同样公式
  TMUL(c2, sum2, mean_t);
  TSUB(c2, c2, sum1);
  TMUL(c3, rstd_t, rstd_t);
  TMUL(c3, c3, rstd_t);
  TMUL(c2, c2, c3);
  TMULS(c2, c2, s);

  TMUL(c3, c2, mean_t);
  TMULS(c3, c3, -1.0f);
  TMUL(sum1, sum2, rstd_t);
  TMULS(sum1, sum1, s);
  TSUB(c3, c3, sum1);

  TSTORE(gc2, c2);
  TSTORE(gc3, c3);
}

// ---------------------------------------------------------------------------
// Step 3: dX for one (n, c) using stored c2/c3 and rstd*gamma
//
// Torch: gpu_kernel 元素级 (可选先算 c1)
//   block = 128
//   vt    = 4 (fp16/bf16) / 2 (fp32+)
//   grid  = ceil(numel / (128 * vt))   // numel = N*C*HxW
//   线程: 线性下标覆盖全部元素；c2/c3 按 (n,g) 广播
// 本函数一次处理一个 (n,c) 的整段 HxW（Tile 覆盖空间维）
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dx_nc(dtype *dy, dtype *x, dtype *gamma, float *rstd, float *c2_buf,
                  float *c3_buf, dtype *dx, int64_t N, int64_t C, int64_t G,
                  int64_t D, int64_t HxW, int64_t tile_hw, int64_t n,
                  int64_t c) {
  const int64_t g = c / D;
  const int64_t ng = n * G + g;
  gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
  gm_f gc2(c2_buf + ng, static_cast<int>(N * G), 1);
  gm_f gc3(c3_buf + ng, static_cast<int>(N * G), 1);

  using scalar_tile = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1>;
  scalar_tile rstd_t, c1, c2, c3;

  TLOAD(rstd_t, grstd);
  TLOAD(c2, gc2);
  TLOAD(c3, gc3);

  // Torch 可选 c1 预计算同为 gpu_kernel block=128；此处 c1 = rstd*gamma[c]
  // Gamma is already one value: convert directly, without a reduction.
  {
    gm_h gg(gamma + c, 1, 1);
    Tile<Location::Vec, dtype, 32, 2, BLayout::CubeM32, 1, 1> hg;
    Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1> gv;
    TLOAD(hg, gg);
    TCVT(gv, hg);
    TMUL(c1, gv, rstd_t);
  }

  const int64_t base = (n * C + c) * HxW;
  for (int64_t hw0 = 0; hw0 < HxW; hw0 += tile_hw) {
    const size_t active_hw =
        static_cast<size_t>((hw0 + tile_hw <= HxW) ? tile_hw : (HxW - hw0));
    const int64_t offset = base + hw0;
    gm_h gdy(dy + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    gm_h gx(x + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    gm_h gdx(dx + offset, static_cast<int>(N * C), static_cast<int>(HxW));
    tile_h h0(1, active_hw);
    tile_f x_f(1, active_hw);
    tile_f dy_f(1, active_hw);
    tile_f dx_f(1, active_hw);
    tile_f tmp(1, active_hw);
    TLOAD(h0, gx);
    TCVT(x_f, h0);
    TLOAD(h0, gdy);
    TCVT(dy_f, h0);
    TROWEXPANDMUL(dx_f, dy_f, c1);
    TROWEXPANDMUL(tmp, x_f, c2);
    TADD(dx_f, dx_f, tmp);
    TROWEXPANDADD(dx_f, dx_f, c3);
    TCVT(h0, dx_f);
    TSTORE(gdx, h0);
  }
}

// Spatial block: one channel, physical [1,256], valid [1,tile_hw].
template <typename dtype>
inline void spatial_block(dtype *dy, dtype *x, float *ds, float *db, int64_t C,
                          int64_t H, int64_t n, int64_t c,
                          int64_t tile_hw) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 1, 256, BLayout::CubeM32, -1, -1>;
  using TF = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, -1>;
  using TV = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, 1>;
  using Scalar = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1>;
  Scalar sa, ba, loaded;
  TV cur(1);
  GF gs(ds + n * C + c, 1, 1), gb(db + n * C + c, 1, 1);
  // This (n,c) output slot is private to its PE until this stage finishes.
  TEXPANDS(sa, 0.0f);
  TEXPANDS(ba, 0.0f);
  for (int64_t h = 0; h < H; h += tile_hw) {
    const int64_t cols = H - h < tile_hw ? H - h : tile_hw;
    GM gx(x + (n * C + c) * H + h, 1, H),
        gy(dy + (n * C + c) * H + h, 1, H);
    TH v(1, cols);
    TF xf(1, cols), yf(1, cols), prod(1, cols);
    TLOAD(v, gx);
    TCVT(xf, v);
    TLOAD(v, gy);
    TCVT(yf, v);
    TMUL(prod, xf, yf);
    TROWSUM(cur, prod);
    TSTORE(gs, cur);
    TLOAD(loaded, gs);
    TADD(sa, sa, loaded);
    TROWSUM(cur, yf);
    TSTORE(gb, cur);
    TLOAD(loaded, gb);
    TADD(ba, ba, loaded);
  }
  TSTORE(gs, sa);
  TSTORE(gb, ba);
}

// Stage 3, HxW = fa*Cols: one (n,c) folded to [fa,Cols] (one TLOAD per tensor).
// Prefer this over [1,256] strips so SoC latency is paid once per channel, not
// once per 256-wide strip (HxW=2048 → 8x fewer TLOAD round-trips).
template <typename dtype, int Cols>
__attribute__((always_inline)) inline void
dx_fold(dtype *dy, dtype *x, dtype *gamma, float *rstd, float *c2, float *c3,
        dtype *dx, int64_t C, int64_t G, int64_t D, int64_t n, int64_t c,
        int64_t fa) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using T = Tiles<dtype, Cols>;
  const int64_t g = c / D;
  const int64_t off = (n * C + c) * fa * Cols;
  GM gx(x + off, fa, Cols), gy(dy + off, fa, Cols), go(dx + off, fa, Cols);
  GFloat gm_r(rstd + n * G + g, 1, 1), gm_c2(c2 + n * G + g, 1, 1),
      gm_c3(c3 + n * G + g, 1, 1);
  GM gg(gamma + c, 1, 1);
  typename T::H v(fa, Cols);
  typename T::F xf(fa, Cols), yf(fa, Cols), out(fa, Cols);
  typename T::V ones(fa), v1(fa), v2(fa), v3(fa);
  typename T::Scalar rs, s2, s3, c1;
  Tile<Location::Vec, dtype, 32, 2, BLayout::CubeM32, 1, 1> hg;
  Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1> gv;
  TLOAD(rs, gm_r);
  TLOAD(s2, gm_c2);
  TLOAD(s3, gm_c3);
  TLOAD(hg, gg);
  TCVT(gv, hg);
  TMUL(c1, gv, rs);
  TEXPANDS(ones, 1.0f);
  TCOLEXPANDMUL(v1, ones, c1);
  TCOLEXPANDMUL(v2, ones, s2);
  TCOLEXPANDMUL(v3, ones, s3);
  TLOAD(v, gx);
  TCVT(xf, v);
  TLOAD(v, gy);
  TCVT(yf, v);
  TROWEXPANDMUL(out, yf, v1);
  TROWEXPANDMUL(xf, xf, v2);
  TADD(out, out, xf);
  TROWEXPANDADD(out, out, v3);
  TCVT(v, out);
  TSTORE(go, v);
}

// dx within one group: strip along HxW with up to 32 contiguous channels.
// Prefer when HxW does not fold into [fa<=32, 2^k]; Config caps tile_hw at 256.
template <typename dtype>
inline void dx_block(dtype *dy, dtype *x, dtype *gamma, float *rstd, float *c2,
                     float *c3, dtype *dx, int64_t C, int64_t G, int64_t H,
                     int64_t n, int64_t g, int64_t c, int64_t rows,
                     int64_t tile_hw) {
  using GH = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 32, 256, BLayout::CubeM32, -1, -1>;
  using TF = Tile<Location::Vec, float, 32, 256, BLayout::CubeM32, -1, -1>;
  using TV = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, -1, 1>;
  using SV = Tile<Location::Vec, float, 1, 1, BLayout::CubeM32, -1, 1>;
  Tile<Location::Vec, dtype, 32, 1, BLayout::CubeM32, -1, 1> gh(rows);
  GH gm_gamma(gamma + c, rows, 1);
  GF gm_r(rstd + n * G + g, 1, 1), gm_c2(c2 + n * G + g, 1, 1),
      gm_c3(c3 + n * G + g, 1, 1);
  SV rs(1), s2(1), s3(1);
  TV gv(rows), v2(rows), v3(rows), ones(rows);
  TLOAD(gh, gm_gamma);
  TCVT(gv, gh);
  TLOAD(rs, gm_r);
  TLOAD(s2, gm_c2);
  TLOAD(s3, gm_c3);
  TCOLEXPANDMUL(gv, gv, rs);
  TEXPANDS(ones, 1.0f);
  TCOLEXPANDMUL(v2, ones, s2);
  TCOLEXPANDMUL(v3, ones, s3);
  for (int64_t h = 0; h < H; h += tile_hw) {
    const int64_t cols = H - h < tile_hw ? H - h : tile_hw;
    GH gx(x + (n * C + c) * H + h, rows, H),
        gy(dy + (n * C + c) * H + h, rows, H);
    GH go(dx + (n * C + c) * H + h, rows, H);
    TH v(rows, cols);
    TF xf(rows, cols), yf(rows, cols), out(rows, cols);
    TLOAD(v, gx);
    TCVT(xf, v);
    TLOAD(v, gy);
    TCVT(yf, v);
    TROWEXPANDMUL(out, yf, gv);
    TROWEXPANDMUL(xf, xf, v2);
    TADD(out, out, xf);
    TROWEXPANDADD(out, out, v3);
    TCVT(v, out);
    TSTORE(go, v);
  }
}

template <typename dtype, int Rows, int Cols>
inline void gamma_beta_block(float *ds, float *db, float *mean, float *rstd,
                             dtype *dg, dtype *dbeta, int64_t N, int64_t C,
                             int64_t G, int64_t D, int64_t g, int64_t d,
                             int64_t rows, int64_t cols) {
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using GH = global_tensor<dtype, RowMajor<-1, -1>>;
  using TF = Tile<Location::Vec, float, Rows, Cols, BLayout::CubeM32, -1, -1>;
  using TH = Tile<Location::Vec, dtype, Rows, Cols, BLayout::CubeM32, -1, -1>;
  using TV = Tile<Location::Vec, float, Rows, 1, BLayout::CubeM32, -1, 1>;
  TF sf(rows, cols), bf(rows, cols), t(rows, cols), ga(rows, cols),
      ba(rows, cols);
  TV m(rows), r(rows);
  TEXPANDS(ga, 0.0f);
  TEXPANDS(ba, 0.0f);
  for (int64_t n = 0; n < N; ++n) {
    GF gs(ds + n * C + g * D + d, rows, D), gb(db + n * C + g * D + d, rows, D);
    GF gm(mean + n * G + g, rows, 1), gr(rstd + n * G + g, rows, 1);
    TLOAD(sf, gs);
    TLOAD(bf, gb);
    TLOAD(m, gm);
    TLOAD(r, gr);
    TADD(ba, ba, bf);
    TROWEXPANDMUL(t, bf, m);
    TSUB(t, sf, t);
    TROWEXPANDMUL(t, t, r);
    TADD(ga, ga, t);
  }
  TH h(rows, cols);
  GH gg(dg + g * D + d, rows, D), gb(dbeta + g * D + d, rows, D);
  TCVT(h, ga);
  TSTORE(gg, h);
  TCVT(h, ba);
  TSTORE(gb, h);
}

// Tiling: N,C,G,H,reduce_hw,reduce_c,dx_hw,dx_c,gb_d,gb_g.
struct Config {
  int64_t N, C, G, H, rh, rc, dh, dc, bd, bg, D;
  explicit Config(const int64_t *t)
      : N(t[0]), C(t[1]), G(t[2]), H(t[3]), rh(t[4]), rc(t[5]), dh(t[6]),
        dc(t[7]), bd(t[8]), bg(t[9]), D(G > 0 ? C / G : 0) {
    // Keep M32 data carriers <=32 KiB; strip loops cover the remainder.
    if (rh > 256) rh = 256;
    if (dh > 256) dh = 256;
    if (bd > 256) bd = 256;
  }
  bool valid() const {
    return N > 0 && C > 0 && G > 0 && H > 0 && C % G == 0 && rh > 0 &&
           rh <= 512 && rc == 1 && dh > 0 && dh <= 8192 && dc > 0 &&
           dc <= 32 && (dc == 1 || dh <= 256) && bd > 0 && bd <= 8192 &&
           bg > 0 && bg <= 32 && (bg == 1 || bd <= 256);
  }
};
} // namespace gn_grad

template <typename dtype, int peNum>
__attribute__((noinline)) void
group_norm_grad_dynamic(dtype *dy, dtype *x, float *mean, float *rstd,
                        dtype *gamma, const int64_t *tiling, float *workspace,
                        dtype *dx, dtype *dgamma, dtype *dbeta) {
  static_assert(peNum == 4);
  const gn_grad::Config t(tiling);
  const uint32_t tid = get_thread_idx();
  if (!t.valid() || tid >= peNum)
    return;
  // One PE owns a complete group range across every batch and all stages.
  // gamma/beta only consume ds/db written by this PE; no cross-PE barrier.
  const int64_t groups_per_pe = (t.G + peNum - 1) / peNum;
  const int64_t group_begin = tid * groups_per_pe;
  const int64_t group_end =
      group_begin + groups_per_pe < t.G ? group_begin + groups_per_pe : t.G;
  float *ds = workspace, *db = workspace + t.N * t.C;
  float *c2 = workspace + 2 * t.N * t.C, *c3 = c2 + t.N * t.G;
  // Stage 1: spatial, R = HxW.
  {
    const int64_t c_begin = group_begin * t.D, c_end = group_end * t.D;
    const int64_t fr = gn_grad::fold_cols(t.H);
    if (gn_grad::is_tile_cols(t.H)) {
      gn_grad::with_cols(t.H, [&]<int Cols>() {
        for (int64_t n = 0; n < t.N; ++n)
          for (int64_t c = c_begin; c < c_end; c += 32)
            gn_grad::spatial_rows<dtype, Cols>(dy, x, ds, db, t.C, n, c,
                                               c_end - c < 32 ? c_end - c : 32);
      });
    } else if (fr > 0) {
      gn_grad::with_cols(fr, [&]<int Cols>() {
        for (int64_t n = 0; n < t.N; ++n)
          for (int64_t c = c_begin; c < c_end; ++c)
            gn_grad::spatial_fold<dtype, Cols>(dy, x, ds, db, t.C, n, c,
                                               t.H / fr);
      });
    } else {
      for (int64_t n = 0; n < t.N; ++n)
        for (int64_t c = c_begin; c < c_end; ++c)
          gn_grad::spatial_block(dy, x, ds, db, t.C, t.H, n, c, t.rh);
    }
  }
  // Stage 2: fused parameters, R = D.
  {
    const float s = 1.0f / static_cast<float>(t.D * t.H);
    const int64_t fr = gn_grad::fold_cols(t.D);
    if (gn_grad::is_tile_cols(t.D)) {
      gn_grad::with_cols(t.D, [&]<int Cols>() {
        for (int64_t n = 0; n < t.N; ++n)
          for (int64_t g = group_begin; g < group_end; g += 32)
            gn_grad::fused_params_rows<dtype, Cols>(
                gamma, mean, rstd, ds, db, c2, c3, t.C, t.G, n, g,
                group_end - g < 32 ? group_end - g : 32, s);
      });
    } else if (fr > 0) {
      gn_grad::with_cols(fr, [&]<int Cols>() {
        for (int64_t n = 0; n < t.N; ++n)
          for (int64_t g = group_begin; g < group_end; ++g)
            gn_grad::fused_params_fold<dtype, Cols>(gamma, mean, rstd, ds, db,
                                                    c2, c3, t.C, t.G, n, g,
                                                    t.D / fr, s);
      });
    } else {
      using GH = global_tensor<dtype, RowMajor<-1, -1>>;
      using GF = global_tensor<float, RowMajor<-1, -1>>;
      using TH = Tile<Location::Vec, dtype, 1, 256, BLayout::CubeM32, -1, -1>;
      using TF = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, -1>;
      using TV = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, 1>;
      for (int64_t n = 0; n < t.N; ++n)
        for (int64_t g = group_begin; g < group_end; ++g)
          gn_grad::fused_params_group<dtype, GH, GF, TH, TF, TV>(
              gamma, mean, rstd, ds, db, c2, c3, t.N, t.C, t.G, t.D, n, g, s);
    }
  }
  // Stage 3: dx. Prefer folded [fa,Cols] TLOADs when HxW packs into one Tile
  // so each channel pays one Soft latency, not HxW/256 strip round-trips.
  {
    using GH = global_tensor<dtype, RowMajor<-1, -1>>;
    using GF = global_tensor<float, RowMajor<-1, -1>>;
    using TH = Tile<Location::Vec, dtype, 1, 256, BLayout::CubeM32, -1, -1>;
    using TF = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, -1>;
    using TV = Tile<Location::Vec, float, 1, 256, BLayout::CubeM32, -1, 1>;
    const int64_t fr = gn_grad::fold_cols(t.H);
    const int64_t fa = fr > 0 ? t.H / fr : 0;
    if (fr > 0 && fa > 0 && fa <= 32) {
      gn_grad::with_cols(fr, [&]<int Cols>() {
        for (int64_t batch = 0; batch < t.N; ++batch)
          for (int64_t g = group_begin; g < group_end; ++g)
            for (int64_t d = 0; d < t.D; ++d)
              gn_grad::dx_fold<dtype, Cols>(dy, x, gamma, rstd, c2, c3, dx, t.C,
                                            t.G, t.D, batch, g * t.D + d, fa);
      });
    } else {
      for (int64_t batch = 0; batch < t.N; ++batch)
        for (int64_t ng = batch * t.G + group_begin;
             ng < batch * t.G + group_end; ++ng) {
          const int64_t n = ng / t.G, g = ng % t.G;
          for (int64_t d = 0; d < t.D; d += t.dc) {
            const int64_t rows = t.D - d < t.dc ? t.D - d : t.dc;
            if (t.dc > 1)
              gn_grad::dx_block(dy, x, gamma, rstd, c2, c3, dx, t.C, t.G, t.H,
                                n, g, g * t.D + d, rows, t.dh);
            else
              gn_grad::dx_nc<dtype, GH, GF, TH, TF, TV>(
                  dy, x, gamma, rstd, c2, c3, dx, t.N, t.C, t.G, t.D, t.H,
                  t.dh, n, g * t.D + d);
          }
        }
    }
  }
  // Stage 4: gamma/beta.
  {
    const int64_t od = (t.D + t.bd - 1) / t.bd;
    for (int64_t g = group_begin; g < group_end; g += t.bg)
      for (int64_t di = 0; di < od; ++di) {
        const int64_t d = di * t.bd;
        const int64_t rows = group_end - g < t.bg ? group_end - g : t.bg,
                      cols = t.D - d < t.bd ? t.D - d : t.bd;
        if (cols == 4)
          gn_grad::gamma_beta_block<dtype, 32, 4>(
              workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N,
              t.C, t.G, t.D, g, d, rows, cols);
        else if (cols == 2)
          gn_grad::gamma_beta_block<dtype, 32, 2>(
              workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N,
              t.C, t.G, t.D, g, d, rows, cols);
        else if (t.bd <= 256)
          gn_grad::gamma_beta_block<dtype, 32, 256>(
              workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N,
              t.C, t.G, t.D, g, d, rows, cols);
        else
          gn_grad::gamma_beta_block<dtype, 1, 256>(
              workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N,
              t.C, t.G, t.D, g, d, rows, cols);
      }
  }
}
#endif
