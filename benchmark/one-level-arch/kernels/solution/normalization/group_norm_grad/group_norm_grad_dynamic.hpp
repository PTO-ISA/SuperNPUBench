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
// Data Tiles use 32 KiB FP32 / 16 KiB FP16. Row-reduction sources <=2 KiB.
// Spatial, parameter and dx kernels share (n,g) ownership.
// The test calls all four stages in order without an inter-stage PE barrier.
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

namespace gn_grad {

constexpr int64_t workspace_elems(int64_t N, int64_t C, int64_t G) {
  return 2 * N * C + 2 * N * G;
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
  tile_v mean_t(1), rstd_t(1), sum1(1), sum2(1), c2(1), c3(1), partial(1);
  TLOAD(mean_t, gmean);
  TLOAD(rstd_t, grstd);
  TEXPANDS(sum1, 0.0f);
  TEXPANDS(sum2, 0.0f);
  for (int64_t d0 = 0; d0 < D; d0 += 512) {
    const size_t vd = D - d0 < 512 ? D - d0 : 512;
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
    TADD(sum1, sum1, partial);
    TMUL(t0, db_f, gamma_f);
    TROWSUM(partial, t0);
    TADD(sum2, sum2, partial);
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

  tile_v rstd_t(1);
  tile_v c1(1);
  tile_v c2(1);
  tile_v c3(1);

  TLOAD(rstd_t, grstd);
  TLOAD(c2, gc2);
  TLOAD(c3, gc3);

  // Torch 可选 c1 预计算同为 gpu_kernel block=128；此处 c1 = rstd*gamma[c]
  // Convert gamma through matching small Tiles before reducing to one value.
  {
    gm_h gg(gamma + c, 1, 1);
    Tile<Location::Vec, dtype, 1, 512, BLayout::RowMajor, -1, -1> hg(1, 1);
    Tile<Location::Vec, float, 1, 512, BLayout::RowMajor, -1, -1> gf(1, 1);
    tile_v gv(1);
    TLOAD(hg, gg);
    TCVT(gf, hg);
    TROWSUM(gv, gf);
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

// Spatial block: one channel, physical [1,512], valid [1,tile_hw].
template <typename dtype>
inline void spatial_block(dtype *dy, dtype *x, float *ds, float *db, int64_t C,
                          int64_t H, int64_t n, int64_t c,
                          int64_t tile_hw) {
  using GM = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 1, 512, BLayout::RowMajor, -1, -1>;
  using TF = Tile<Location::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
  using TV = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor, -1, 1>;
  TV sa(1), ba(1), cur(1);
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
    TADD(sa, sa, cur);
    TROWSUM(cur, yf);
    TADD(ba, ba, cur);
  }
  GF gs(ds + n * C + c, 1, 1), gb(db + n * C + c, 1, 1);
  TSTORE(gs, sa);
  TSTORE(gb, ba);
}

// dx within one group: 32x256 FP32 data Tile = 32 KiB.
template <typename dtype>
inline void dx_block(dtype *dy, dtype *x, dtype *gamma, float *rstd, float *c2,
                     float *c3, dtype *dx, int64_t C, int64_t G, int64_t H,
                     int64_t n, int64_t g, int64_t c, int64_t rows,
                     int64_t tile_hw) {
  using GH = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 32, 256, BLayout::RowMajor, -1, -1>;
  using TF = Tile<Location::Vec, float, 32, 256, BLayout::RowMajor, -1, -1>;
  using TV = Tile<Location::Vec, float, 32, 1, BLayout::RowMajor, -1, 1>;
  using SV = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor, -1, 1>;
  Tile<Location::Vec, dtype, 32, 16, BLayout::RowMajor, -1, -1> gh(rows, 1);
  Tile<Location::Vec, float, 32, 16, BLayout::RowMajor, -1, -1> gf(rows, 1);
  GH gm_gamma(gamma + c, rows, 1);
  GF gm_r(rstd + n * G + g, 1, 1), gm_c2(c2 + n * G + g, 1, 1),
      gm_c3(c3 + n * G + g, 1, 1);
  SV rs(1), s2(1), s3(1);
  TV gv(rows), v2(rows), v3(rows), ones(rows);
  TLOAD(gh, gm_gamma);
  TCVT(gf, gh);
  TROWSUM(gv, gf);
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
  using TF = Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor, -1, -1>;
  using TH = Tile<Location::Vec, dtype, Rows, Cols, BLayout::RowMajor, -1, -1>;
  using TV = Tile<Location::Vec, float, Rows, 1, BLayout::RowMajor, -1, 1>;
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
        dc(t[7]), bd(t[8]), bg(t[9]), D(G > 0 ? C / G : 0) {}
  bool valid() const {
    return N > 0 && C > 0 && G > 0 && H > 0 && C % G == 0 && rh > 0 &&
           rh <= 512 && rc == 1 && dh > 0 && dh <= 8192 && dc > 0 &&
           dc <= 32 && (dc == 1 || dh <= 256) && bd > 0 && bd <= 8192 &&
           bg > 0 && bg <= 32 && (bg == 1 || bd <= 256);
  }
};
} // namespace gn_grad

template <typename dtype, int peNum>
__attribute__((noinline)) void group_norm_grad_spatial(dtype *dy, dtype *x,
                                                       const int64_t *tiling,
                                                       float *workspace) {
  static_assert(peNum == 4);
  gn_grad::Config t(tiling);
  const uint32_t tid = get_thread_idx();
  if (!t.valid() || tid >= peNum)
    return;
  for (int64_t ng = tid; ng < t.N * t.G; ng += peNum) {
    const int64_t n = ng / t.G, g = ng % t.G;
    for (int64_t d = 0; d < t.D; d += t.rc)
      gn_grad::spatial_block(dy, x, workspace, workspace + t.N * t.C, t.C, t.H,
                             n, g * t.D + d, t.rh);
  }
}

template <typename dtype, int peNum>
__attribute__((noinline)) void
group_norm_grad_fused_params(dtype *gamma, float *mean, float *rstd,
                             const int64_t *tiling, float *workspace) {
  static_assert(peNum == 4);
  gn_grad::Config t(tiling);
  const uint32_t tid = get_thread_idx();
  if (!t.valid() || tid >= peNum)
    return;
  using GH = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 1, 512, BLayout::RowMajor, -1, -1>;
  using TF = Tile<Location::Vec, float, 1, 512, BLayout::RowMajor, -1, -1>;
  using TV = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor, -1, 1>;
  float *c2 = workspace + 2 * t.N * t.C, *c3 = c2 + t.N * t.G;
  for (int64_t ng = tid; ng < t.N * t.G; ng += peNum)
    gn_grad::fused_params_group<dtype, GH, GF, TH, TF, TV>(
        gamma, mean, rstd, workspace, workspace + t.N * t.C, c2, c3, t.N, t.C,
        t.G, t.D, ng / t.G, ng % t.G, 1.0f / static_cast<float>(t.D * t.H));
}

template <typename dtype, int peNum>
__attribute__((noinline)) void
group_norm_grad_dx(dtype *dy, dtype *x, dtype *gamma, float *rstd,
                   const int64_t *tiling, float *workspace, dtype *dx) {
  static_assert(peNum == 4);
  gn_grad::Config t(tiling);
  const uint32_t tid = get_thread_idx();
  if (!t.valid() || tid >= peNum)
    return;
  using GH = global_tensor<dtype, RowMajor<-1, -1>>;
  using GF = global_tensor<float, RowMajor<-1, -1>>;
  using TH = Tile<Location::Vec, dtype, 1, 8192, BLayout::RowMajor, -1, -1>;
  using TF = Tile<Location::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1>;
  using TV = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor, -1, 1>;
  float *c2 = workspace + 2 * t.N * t.C, *c3 = c2 + t.N * t.G;
  for (int64_t ng = tid; ng < t.N * t.G; ng += peNum) {
    const int64_t n = ng / t.G, g = ng % t.G;
    for (int64_t d = 0; d < t.D; d += t.dc) {
      const int64_t rows = t.D - d < t.dc ? t.D - d : t.dc;
      if (t.dc > 1)
        gn_grad::dx_block(dy, x, gamma, rstd, c2, c3, dx, t.C, t.G, t.H, n, g,
                          g * t.D + d, rows, t.dh);
      else
        gn_grad::dx_nc<dtype, GH, GF, TH, TF, TV>(dy, x, gamma, rstd, c2, c3,
                                                  dx, t.N, t.C, t.G, t.D, t.H,
                                                  t.dh, n, g * t.D + d);
    }
  }
}

template <typename dtype, int peNum>
__attribute__((noinline)) void
group_norm_grad_gamma_beta(float *mean, float *rstd, const int64_t *tiling,
                           float *workspace, dtype *dgamma, dtype *dbeta) {
  static_assert(peNum == 4);
  gn_grad::Config t(tiling);
  const uint32_t tid = get_thread_idx();
  if (!t.valid() || tid >= peNum)
    return;
  const int64_t og = (t.G + t.bg - 1) / t.bg, od = (t.D + t.bd - 1) / t.bd;
  for (int64_t task = tid; task < og * od; task += peNum) {
    const int64_t g = task / od * t.bg, d = task % od * t.bd;
    const int64_t rows = t.G - g < t.bg ? t.G - g : t.bg,
                  cols = t.D - d < t.bd ? t.D - d : t.bd;
    if (t.bd <= 256)
      gn_grad::gamma_beta_block<dtype, 32, 256>(
          workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N, t.C,
          t.G, t.D, g, d, rows, cols);
    else
      gn_grad::gamma_beta_block<dtype, 1, 8192>(
          workspace, workspace + t.N * t.C, mean, rstd, dgamma, dbeta, t.N, t.C,
          t.G, t.D, g, d, rows, cols);
  }
}
#endif
