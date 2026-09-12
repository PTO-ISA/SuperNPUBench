// =============================================================================
// group_norm_grad_1d.hpp — GroupNorm backward, HxW == 1 (one-level PTO)
// =============================================================================
//
// Matches PyTorch GroupNorm1dBackward
// (aten/src/ATen/native/cuda/group_norm_kernel.cu):
//   Stage A1  reduce → c2/c3  (per n,g)
//   Stage A2  dX = (rstd*gamma)*dY + c2*X + c3
//   Stage B   dgamma / dbeta
//
// Layout: X/dY/dX [N,C]; mean/rstd [N,G] fp32; gamma/dgamma/dbeta [C]
// All pointers required, including caller-owned workspace of 2*N*G floats.
// Each PE writes its own parameters before consuming them in Stage A2.
//
// tiling[7] = {N, C, G, tile_d, tile_g, gb_tile_d, gb_tile_g}
//   tile_d <= 0 → min(D, tD); channel R-split on dgamma/dbeta when D > tile_d.
//   Stage A and B split arbitrary D into tiles, including partial tails.
//
// Data tiles: 8192 columns, FP32 32 KiB and FP16 16 KiB.
// Parameter reduction uses 512-column strips (FP32 2 KiB).
// Reduction/broadcast outputs retain physical Columns=1.
// Reduce and dX are separate passes so large tiles do not stay live across both.
//
// Torch CUDA launch 总览 (NVIDIA, warp=32; HxW==1 特化):
//   A1 Compute1dBackwardFusedParamsCUDAKernel
//     grid=dim3(N,G), block=(D<512)?32:512
//   A2 dX gpu_kernel
//     block=128, vt=4(fp16)/2(fp32), grid=ceil(N*C/(128*vt))
//   B  GammaBeta1dBackwardCUDAKernel1/2
//     N<=128: grid=ceil(C/256), block=256
//     N>128:  grid=ceil(C/32),  block=dim3(32,16)
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP
#define SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace gn_grad_1d {
// Caller-owned GM workspace [2,N*G]: all c2 values, then all c3 values.
inline int64_t workspace_elems(int64_t N, int64_t G) { return 2 * N * G; }

// ---------------------------------------------------------------------------
// Stage A1: channel reduce → c2/c3 for one (n, g)
//   scratch points to c2[n,g]; c3[n,g] is at scratch + N*G.
//
// Torch: Compute1dBackwardFusedParamsCUDAKernel
//   grid  = dim3(N, G)     // blockIdx.x=n, blockIdx.y=g；本函数 = 其中一个
//   block = (D < 512) ? 32 : 512
//   线程: threadIdx.x 沿 i∈[0,D) stride，读 dY/X/gamma 累加后 block reduce
//   → sum1=Σ dY*X*gamma, sum2=Σ dY*gamma → c2,c3
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void fused_params_group(dtype *dy, dtype *x, float *mean, float *rstd,
                               dtype *gamma, float *scratch, int64_t N,
                               int64_t C, int64_t G, int64_t D, int64_t n,
                               int64_t g, float s, int64_t tile_d) {
    const int64_t ng = n * G + g;
    const int64_t c0 = g * D;
    const int64_t offset = n * C + c0;



    gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gc2(scratch + 0, 1, 1);
    gm_f gc3(scratch + N * G, 1, 1);

    tile_v sum1(1), sum2(1);
    TEXPANDS(sum1, 0.0f);
    TEXPANDS(sum2, 0.0f);
    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(D - d0 < tile_d ? D - d0 : tile_d);
        gm_h gdy(dy + offset + d0, static_cast<int>(N), static_cast<int>(C));
        gm_h gx(x + offset + d0, static_cast<int>(N), static_cast<int>(C));
        gm_h gg(gamma + c0 + d0, 1, static_cast<int>(C));
        tile_h h(1, vd);
        tile_f xf(1, vd), dyf(1, vd), gf(1, vd), prod(1, vd);
        tile_v partial1(1), partial2(1);
        TLOAD(h, gx);
        TCVT(xf, h);
        TLOAD(h, gdy);
        TCVT(dyf, h);
        TLOAD(h, gg);
        TCVT(gf, h);
        TMUL(prod, dyf, gf);
        TROWSUM(partial2, prod);
        TMUL(prod, prod, xf);
        TROWSUM(partial1, prod);
        TADD(sum1, sum1, partial1);
        TADD(sum2, sum2, partial2);
    }
    tile_v mean_t(1), rstd_t(1), c2(1), c3(1);
    TLOAD(mean_t, gmean);
    TLOAD(rstd_t, grstd);

    // c2 = (sum2*mean - sum1) * rstd^3 * s   （归约后标量，通常 thread0 写）
    TMUL(c2, sum2, mean_t);
    TSUB(c2, c2, sum1);
    TMUL(c3, rstd_t, rstd_t);
    TMUL(c3, c3, rstd_t);
    TMUL(c2, c2, c3);
    TMULS(c2, c2, s);

    // c3 = -c2*mean - sum2*rstd*s
    TMUL(c3, c2, mean_t);
    TMULS(c3, c3, -1.0f);
    TMUL(sum1, sum2, rstd_t);
    TMULS(sum1, sum1, s);
    TSUB(c3, c3, sum1);

    TSTORE(gc2, c2);
    TSTORE(gc3, c3);
}

// ---------------------------------------------------------------------------
// Stage A2: dX for one (n, g) from spilled c2/c3
//
// Torch: gpu_kernel 元素级
//   block = 128
//   vt    = 4 (fp16/bf16) / 2 (fp32+)
//   grid  = ceil(N*C / (128*vt))
//   线程: 线性下标覆盖 [N,C]；c2/c3 按 (n,g) 广播到组内通道
// 本函数一次写完一组 D 个通道（HxW=1）
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dx_group(dtype *dy, dtype *x, float *rstd, dtype *gamma,
                     float *scratch, dtype *dx, int64_t N, int64_t C,
                     int64_t G, int64_t D, int64_t n, int64_t g, int64_t tile_d) {
    const int64_t ng = n * G + g;
    const int64_t c0 = g * D;
    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const int64_t offset = n * C + c0 + d0;
        const size_t active_d = static_cast<size_t>(D - d0 < tile_d ? D - d0 : tile_d);

        gm_h gdy(dy + offset, static_cast<int>(N), static_cast<int>(C));
        gm_h gx(x + offset, static_cast<int>(N), static_cast<int>(C));
        gm_h gdx(dx + offset, static_cast<int>(N), static_cast<int>(C));
        gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
        gm_f gc2(scratch + 0, 1, 1);
        gm_f gc3(scratch + N * G, 1, 1);

        tile_h h0(1, active_d);
        tile_h h1(1, active_d);
        tile_f x_f(1, active_d);
        tile_f dy_f(1, active_d);
        tile_f t0(1, active_d);
        tile_f t1(1, active_d);
        tile_v rstd_t(1);
        tile_v c2(1);
        tile_v c3(1);

        TLOAD(h0, gx);
        TCVT(x_f, h0);
        TLOAD(h0, gdy);
        TCVT(dy_f, h0);
        TLOAD(rstd_t, grstd);
        TLOAD(c2, gc2);
        TLOAD(c3, gc3);

        {
            gm_h gg(gamma + c0 + d0, 1, static_cast<int>(C));
            TLOAD(h1, gg);
            TCVT(t0, h1); // gamma
        }

        // dX = (rstd*gamma)*dY + c2*X + c3
        TROWEXPANDMUL(t1, t0, rstd_t);
        TMUL(t1, t1, dy_f);
        TROWEXPANDMUL(t0, x_f, c2);
        TADD(t1, t1, t0);
        TROWEXPANDADD(t1, t1, c3);

        TCVT(h0, t1);
        TSTORE(gdx, h0);
    }
}

// Small-D block: logical [active_g,D], physical [32,256] FP32 = 32 KiB.
template <typename dtype>
inline void dx_groups(dtype *dy, dtype *x, float *rstd, dtype *gamma,
                      float *workspace, dtype *dx, int64_t N, int64_t C, int64_t G,
                      int64_t D, int64_t n, int64_t g0, int64_t active_g) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using htile = Tile<Location::Vec, dtype, 32, 256, BLayout::RowMajor, -1, -1>;
    using ftile = Tile<Location::Vec, float, 32, 256, BLayout::RowMajor, -1, -1>;
    using vtile = Tile<Location::Vec, float, 32, 1, BLayout::RowMajor, -1, 1>;
    const int64_t ng = n * G + g0;
    const int64_t offset = n * C + g0 * D;
    gm_h gx(x + offset, static_cast<int>(active_g), static_cast<int>(D));
    gm_h gdy(dy + offset, static_cast<int>(active_g), static_cast<int>(D));
    gm_h gg(gamma + g0 * D, static_cast<int>(active_g), static_cast<int>(D));
    gm_h gout(dx + offset, static_cast<int>(active_g), static_cast<int>(D));
    gm_f gr(rstd + ng, static_cast<int>(active_g), 1);
    gm_f gc2(workspace + ng, static_cast<int>(active_g), 1);
    gm_f gc3(workspace + N * G + ng, static_cast<int>(active_g), 1);
    htile h(active_g, D);
    ftile xf(active_g, D), dyf(active_g, D), gf(active_g, D), out(active_g, D);
    vtile r(active_g), c2(active_g), c3(active_g);
    TLOAD(h, gx); TCVT(xf, h);
    TLOAD(h, gdy); TCVT(dyf, h);
    TLOAD(h, gg); TCVT(gf, h);
    TLOAD(r, gr);
    TLOAD(c2, gc2);
    TLOAD(c3, gc3);
    TROWEXPANDMUL(out, gf, r);
    TMUL(out, out, dyf);
    TROWEXPANDMUL(xf, xf, c2);
    TADD(out, out, xf);
    TROWEXPANDADD(out, out, c3);
    TCVT(h, out);
    TSTORE(gout, h);
}

// ---------------------------------------------------------------------------
// Stage B: dbeta — dbeta[c] = Σ_n dY[n,c]
//
// Torch: GammaBeta1dBackwardCUDAKernel1/2（与 dgamma 同 launch）
//   N<=128: grid=ceil(C/256), block=256；每线程一个 c，循环 n
//   N>128:  grid=ceil(C/32),  block=dim3(32,16)
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dbeta_group(dtype *dy, dtype *dbeta, int64_t N, int64_t C,
                        int64_t D, int64_t tile_d, int64_t g) {
    const int64_t c0 = g * D;

    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(
            (d0 + tile_d <= D) ? tile_d : (D - d0));

        tile_h h0(1, vd);
        tile_f dy_f(1, vd);
        tile_f acc(1, vd);
        TEXPANDS(acc, 0.0f);

        // Torch Kernel1: 单线程 for(n) 累加；此处 Tile 一次累加一组通道
        for (int64_t n = 0; n < N; ++n) {
            gm_h gdy(dy + n * C + c0 + d0, static_cast<int>(N),
                     static_cast<int>(C));
            TLOAD(h0, gdy);
            TCVT(dy_f, h0);
            TADD(acc, acc, dy_f);
        }

        gm_h gdb(dbeta + c0 + d0, 1, static_cast<int>(C));
        TCVT(h0, acc);
        TSTORE(gdb, h0);
    }
}

// ---------------------------------------------------------------------------
// Stage B: dgamma — dgamma[c] = Σ_n dY*(X-mean)*rstd
//
// Torch: 与 dbeta 同 Kernel1/2 launch（见上）
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dgamma_group(dtype *dy, dtype *x, float *mean, float *rstd,
                         dtype *dgamma, int64_t N, int64_t C, int64_t G,
                         int64_t D, int64_t tile_d, int64_t g) {
    const int64_t c0 = g * D;

    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(
            (d0 + tile_d <= D) ? tile_d : (D - d0));

        tile_h h0(1, vd);
        tile_f dy_f(1, vd);
        tile_f x_f(1, vd);
        tile_f t0(1, vd);
        tile_f acc(1, vd);
        tile_v mean_t(1);
        tile_v rstd_t(1);
        TEXPANDS(acc, 0.0f);

        for (int64_t n = 0; n < N; ++n) {
            const int64_t ng = n * G + g;
            const int64_t offset = n * C + c0 + d0;

            gm_h gdy(dy + offset, static_cast<int>(N), static_cast<int>(C));
            gm_h gx(x + offset, static_cast<int>(N), static_cast<int>(C));
            gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
            gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);

            TLOAD(h0, gdy);
            TCVT(dy_f, h0);
            TLOAD(h0, gx);
            TCVT(x_f, h0);
            TLOAD(mean_t, gmean);
            TLOAD(rstd_t, grstd);

            TROWEXPANDMUL(t0, x_f, rstd_t);
            TMUL(t0, t0, dy_f);
            TROWEXPANDMUL(x_f, dy_f, mean_t);
            TROWEXPANDMUL(x_f, x_f, rstd_t);
            TSUB(t0, t0, x_f);
            TADD(acc, acc, t0);
        }

        gm_h gdg(dgamma + c0 + d0, 1, static_cast<int>(C));
        TCVT(h0, acc);
        TSTORE(gdg, h0);
    }
}

// Stage B two-dimensional block: rows are groups, columns are channels within
// a group. GM row stride remains D even for a partial channel tile.
template <typename dtype>
inline void gamma_beta_groups(dtype *dy, dtype *x, float *mean, float *rstd,
                              dtype *dgamma, dtype *dbeta, int64_t N, int64_t C,
                              int64_t G, int64_t D, int64_t g0, int64_t d0,
                              int64_t active_g, int64_t active_d) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using ht = Tile<Location::Vec, dtype, 32, 256, BLayout::RowMajor, -1, -1>;
    using ft = Tile<Location::Vec, float, 32, 256, BLayout::RowMajor, -1, -1>;
    using vt = Tile<Location::Vec, float, 32, 1, BLayout::RowMajor, -1, 1>;
    ht h(active_g, active_d);
    ft dyf(active_g, active_d), xf(active_g, active_d), tmp(active_g, active_d);
    ft beta(active_g, active_d), grad(active_g, active_d);
    vt m(active_g), r(active_g);
    TEXPANDS(beta, 0.0f);
    TEXPANDS(grad, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        const int64_t offset = n * C + g0 * D + d0;
        gm_h gdy(dy + offset, static_cast<int>(active_g), static_cast<int>(D));
        gm_h gx(x + offset, static_cast<int>(active_g), static_cast<int>(D));
        gm_f gm(mean + n * G + g0, static_cast<int>(active_g), 1);
        gm_f gr(rstd + n * G + g0, static_cast<int>(active_g), 1);
        TLOAD(h, gdy);
        TCVT(dyf, h);
        TADD(beta, beta, dyf);
        TLOAD(h, gx);
        TCVT(xf, h);
        TLOAD(m, gm);
        TLOAD(r, gr);
        // Preserve the original FP32 operation order for dgamma.
        TROWEXPANDMUL(tmp, xf, r);
        TMUL(tmp, tmp, dyf);
        TROWEXPANDMUL(xf, dyf, m);
        TROWEXPANDMUL(xf, xf, r);
        TSUB(tmp, tmp, xf);
        TADD(grad, grad, tmp);
    }
    gm_h gb(dbeta + g0 * D + d0, static_cast<int>(active_g), static_cast<int>(D));
    gm_h gg(dgamma + g0 * D + d0, static_cast<int>(active_g), static_cast<int>(D));
    TCVT(h, beta);
    TSTORE(gb, h);
    TCVT(h, grad);
    TSTORE(gg, h);
}

} // namespace gn_grad_1d

// Three separate 4PE kernels; tiling = [N, C, G, tile_d, tile_g, gb_tile_d, gb_tile_g].
// Call stages in order with identical PE ownership for parameters and dx.
template <typename dtype, int peNum>
__attribute__((noinline)) void group_norm_grad_1d_fused_params(
    dtype *dy, dtype *x, float *mean, float *rstd,
    dtype *gamma, const int64_t *tiling, float *workspace) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    // TROWSUM source descriptor is limited to 2048 bytes in this model.
    // Keep this reduction strip at 512 FP32 elements; other stages use 32 KiB.
    constexpr int64_t tD = 512;
    constexpr int64_t tV = 1; // row-reduction/broadcast physical Columns=1

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const uint32_t tid = get_thread_idx();
    if (N <= 0 || C <= 0 || G <= 0 || C % G != 0 ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }
    const int64_t D = C / G;
    const int64_t requested_d = tiling[3] > 0 ? tiling[3] : D;
    const int64_t tile_d = requested_d < tD ? requested_d : tD;
    if (tile_d <= 0 || tile_d > tD) {
        return;
    }

    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_v =
        Tile<Location::Vec, float, 1, tV, BLayout::RowMajor, -1, 1>;

    const int64_t tile_g = tiling[4];
    if (tile_g < 1 || tile_g > 32 || (D > 256 && tile_g != 1)) return;
    const int64_t outer_g = (G + tile_g - 1) / tile_g;
    const float s = 1.0f / static_cast<float>(D);
    for (int64_t task = tid; task < N * outer_g; task += peNum) {
        const int64_t n = task / outer_g;
        const int64_t g0 = (task % outer_g) * tile_g;
        const int64_t end_g = g0 + tile_g < G ? g0 + tile_g : G;
        for (int64_t g = g0; g < end_g; ++g) {
            gn_grad_1d::fused_params_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
                dy, x, mean, rstd, gamma, workspace + n * G + g,
                N, C, G, D, n, g, s, tile_d);
        }
    }
}

template <typename dtype, int peNum>
__attribute__((noinline)) void group_norm_grad_1d_dx(
    dtype *dy, dtype *x, float *rstd, dtype *gamma,
    const int64_t *tiling, float *workspace, dtype *dx) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    // FP32 Tile: 32 KiB (8192 columns); FP16: 16 KiB, matching TCVT shape.
    constexpr int64_t tDDtype =
        (32768 + static_cast<int64_t>(sizeof(dtype)) - 1) /
        static_cast<int64_t>(sizeof(dtype));
    constexpr int64_t tD = tDDtype < 8192 ? tDDtype : 8192;
    constexpr int64_t tV = 1; // row-reduction/broadcast physical Columns=1

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const uint32_t tid = get_thread_idx();
    if (N <= 0 || C <= 0 || G <= 0 || C % G != 0 ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }
    const int64_t D = C / G;
    const int64_t tile_d = tiling[3] > 0 ? tiling[3] : (D < tD ? D : tD);
    if (tile_d <= 0 || tile_d > tD) {
        return;
    }

    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_v =
        Tile<Location::Vec, float, 1, tV, BLayout::RowMajor, -1, 1>;

    const int64_t tile_g = tiling[4];
    if (tile_g < 1 || tile_g > 32 || (D > 256 && tile_g != 1)) return;
    const int64_t outer_g = (G + tile_g - 1) / tile_g;
    for (int64_t task = tid; task < N * outer_g; task += peNum) {
        const int64_t n = task / outer_g;
        const int64_t g0 = (task % outer_g) * tile_g;
        const int64_t active_g = G - g0 < tile_g ? G - g0 : tile_g;
        if (D <= 256 && tile_g > 1) {
            gn_grad_1d::dx_groups(dy, x, rstd, gamma, workspace, dx,
                                  N, C, G, D, n, g0, active_g);
        } else {
            gn_grad_1d::dx_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
                dy, x, rstd, gamma, workspace + n * G + g0,
                dx, N, C, G, D, n, g0, tile_d);
        }
    }
}

template <typename dtype, int peNum>
__attribute__((noinline)) void group_norm_grad_1d_gamma_beta(
    dtype *dy, dtype *x, float *mean, float *rstd,
    const int64_t *tiling, dtype *dgamma, dtype *dbeta) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    // FP32 Tile: 32 KiB (8192 columns); FP16: 16 KiB, matching TCVT shape.
    constexpr int64_t tDDtype =
        (32768 + static_cast<int64_t>(sizeof(dtype)) - 1) /
        static_cast<int64_t>(sizeof(dtype));
    constexpr int64_t tD = tDDtype < 8192 ? tDDtype : 8192;
    constexpr int64_t tV = 1; // row-reduction/broadcast physical Columns=1

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const uint32_t tid = get_thread_idx();
    if (N <= 0 || C <= 0 || G <= 0 || C % G != 0 ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }
    const int64_t D = C / G;
    const int64_t tile_d = tiling[5] > 0 ? tiling[5] : (D < tD ? D : tD);
    if (tile_d <= 0 || tile_d > tD) {
        return;
    }

    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tD, BLayout::RowMajor, -1, -1>;
    using tile_v =
        Tile<Location::Vec, float, 1, tV, BLayout::RowMajor, -1, 1>;

    const int64_t tile_g = tiling[6];
    if (tile_g < 1 || tile_g > 32 || (tile_d > 256 && tile_g != 1)) return;
    if (tile_d <= 256) {
        const int64_t outer_g = (G + tile_g - 1) / tile_g;
        const int64_t outer_d = (D + tile_d - 1) / tile_d;
        // Each PE owns complete output blocks; N is reduced locally.
        for (int64_t task = tid; task < outer_g * outer_d; task += peNum) {
            const int64_t g0 = (task / outer_d) * tile_g;
            const int64_t d0 = (task % outer_d) * tile_d;
            const int64_t vg = G - g0 < tile_g ? G - g0 : tile_g;
            const int64_t vd = D - d0 < tile_d ? D - d0 : tile_d;
            gn_grad_1d::gamma_beta_groups(dy, x, mean, rstd, dgamma, dbeta,
                                         N, C, G, D, g0, d0, vg, vd);
        }
        return;
    }
    for (int64_t g = tid; g < G; g += peNum) {
        gn_grad_1d::dbeta_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            dy, dbeta, N, C, D, tile_d, g);
        gn_grad_1d::dgamma_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            dy, x, mean, rstd, dgamma, N, C, G, D, tile_d, g);
    }
}


#endif // SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP
