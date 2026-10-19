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
//   tile_d controls parameter/dx strips; gb_tile_d controls gamma/beta strips.
//   tile_d <= 0 → min(D, tD); channel splitting continues when D exceeds the selected strip width.
//   Stage A and B split arbitrary D into tiles, including partial tails.
//
// Stage A1 reduces R=D in one pass (group_norm_grad_reduce.hpp, Tiles
// <= 32 KiB) when D is a power of two <= 256 or folds to [<=32, 2^k<=256];
// otherwise it falls back to 64-column strips with partial sums spilled
// through GM.
// Stage B puts 32 batches in the Tile rows of one W-channel block (W | D,
// W <= 64), TADDs the blocks and reduces N with one TCOLSUM; blocks are spread
// over the PEs. Odd D keeps the per-group strip path.
// Four PEs share one 256 KiB tile-register pool. Stage A2/B strips stay at 64
// CubeM32 columns (FP32 8 KiB, FP16 4 KiB) with broadcast tiles at Columns=1.
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

#include "group_norm_grad_reduce.hpp"

namespace gn_grad_1d {
// Caller-owned GM workspace. Staged paths: [2,N*G], all c2 values, then all
// c3 values. Fused path: dbeta then dgamma column partials, one [1,D] row per
// (g, 32-batch block).
constexpr int64_t workspace_elems(int64_t N, int64_t C, int64_t G) {
    const int64_t staged = 2 * N * G;
    const int64_t fused = 2 * ((N + 31) / 32) * C;
    return staged > fused ? staged : fused;
}

// Shared shape validation and physical Tile definitions for all three stages.
// CubeM32 always allocates 32 physical rows. 64 columns => 8 KiB FP32.
constexpr int kPhysRows = 32;
constexpr int kPhysCols = 64;
constexpr int64_t kTileBytes = 8192;

template <typename dtype>
constexpr int64_t data_columns() {
    constexpr int64_t dtype_cols = kTileBytes / (32 * sizeof(dtype));
    return dtype_cols < kPhysCols ? dtype_cols : kPhysCols;
}
struct Shape {
    int64_t N, C, G, D;
    explicit Shape(const int64_t *t)
        : N(t[0]), C(t[1]), G(t[2]), D(C / G) {}
};
template <typename dtype, int Rows, int Cols>
struct TileTypes {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h = Tile<Location::Vec, dtype, Rows, Cols, BLayout::CubeM32, -1, -1>;
    using tile_f = Tile<Location::Vec, float, Rows, Cols, BLayout::CubeM32, -1, -1>;
    using tile_v = Tile<Location::Vec, float, Rows, 1, BLayout::CubeM32, -1, 1>;
};

using namespace gn_grad_reduce;

// Stage A1, D = Cols: groups [g0, g0+rows) of batch n as one [rows,D] Tile.
template <typename dtype, int Cols>
inline void fused_params_rows(dtype *dy, dtype *x, float *mean, float *rstd,
                              dtype *gamma, float *workspace, int64_t N,
                              int64_t C, int64_t G, int64_t n, int64_t g0,
                              int64_t rows, float s) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using T = Tiles<dtype, Cols>;
    const int64_t ng = n * G + g0;
    const int64_t offset = n * C + g0 * Cols;
    gm_h gx(x + offset, rows, Cols), gdy(dy + offset, rows, Cols);
    gm_h gg(gamma + g0 * Cols, rows, Cols);
    GFloat gm(mean + ng, rows, 1), gr(rstd + ng, rows, 1);
    GFloat gc2(workspace + ng, rows, 1), gc3(workspace + N * G + ng, rows, 1);
    typename T::H h(rows, Cols);
    typename T::F dyf(rows, Cols), gf(rows, Cols), xf(rows, Cols);
    typename T::W w1(rows), w2(rows);
    TLOAD(h, gdy);
    TCVT(dyf, h);
    TLOAD(h, gg);
    TCVT(gf, h);
    TMUL(dyf, dyf, gf);
    TLOAD(h, gx);
    TCVT(xf, h);
    TROWSUM(w2, dyf);
    TMUL(xf, xf, dyf);
    TROWSUM(w1, xf);
    c2_c3_rows<T>(w1, w2, gm, gr, gc2, gc3, rows, s);
}

// Stage A1, D = fa*Cols: one (n,g) folded to [fa,Cols].
template <typename dtype, int Cols>
inline void fused_params_fold(dtype *dy, dtype *x, float *mean, float *rstd,
                              dtype *gamma, float *workspace, int64_t N,
                              int64_t C, int64_t G, int64_t n, int64_t g,
                              int64_t fa, float s) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using T = Tiles<dtype, Cols>;
    const int64_t ng = n * G + g;
    const int64_t offset = n * C + g * fa * Cols;
    gm_h gx(x + offset, fa, Cols), gdy(dy + offset, fa, Cols);
    gm_h gg(gamma + g * fa * Cols, fa, Cols);
    GFloat gm(mean + ng, 1, 1), gr(rstd + ng, 1, 1);
    GFloat gc2(workspace + ng, 1, 1), gc3(workspace + N * G + ng, 1, 1);
    typename T::H h(fa, Cols);
    typename T::F dyf(fa, Cols), gf(fa, Cols), xf(fa, Cols);
    typename T::W w1(fa), w2(fa);
    TLOAD(h, gdy);
    TCVT(dyf, h);
    TLOAD(h, gg);
    TCVT(gf, h);
    TMUL(dyf, dyf, gf);
    TLOAD(h, gx);
    TCVT(xf, h);
    TROWSUM(w2, dyf);
    TMUL(xf, xf, dyf);
    TROWSUM(w1, xf);
    c2_c3_fold<T>(w1, w2, gm, gr, gc2, gc3, s);
}

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

    using scalar_tile = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1>;
    scalar_tile sum1, sum2, loaded1, loaded2;
    TEXPANDS(sum1, 0.0f);
    TEXPANDS(sum2, 0.0f);
    for (int64_t d0 = 0; d0 < D; d0 += tile_d) {
        const size_t vd = static_cast<size_t>(D - d0 < tile_d ? D - d0 : tile_d);
        gm_h gdy(dy + offset + d0, static_cast<int>(N), static_cast<int>(C));
        gm_h gx(x + offset + d0, static_cast<int>(N), static_cast<int>(C));
        gm_h gg(gamma + c0 + d0, 1, static_cast<int>(C));
        tile_h h(1, vd);
        tile_f xf(1, vd), dyf(1, vd), gf(1, vd), prod(1, vd);
        // PTO #311: TROWSUM dest keeps the source physical columns, ValidCols=1.
        using tile_red = Tile<Location::Vec, float, 32, tile_f::Cols, BLayout::CubeM32, -1, 1>;
        tile_red partial1(1), partial2(1);
        TLOAD(h, gx);
        TCVT(xf, h);
        TLOAD(h, gdy);
        TCVT(dyf, h);
        TLOAD(h, gg);
        TCVT(gf, h);
        TMUL(prod, dyf, gf);
        TROWSUM(partial2, prod);
        // Private (n,g) c2/c3 slots hold partials until final parameters replace them.
        TSTORE(gc3, partial2);
        TMUL(prod, prod, xf);
        TROWSUM(partial1, prod);
        TSTORE(gc2, partial1);
        TLOAD(loaded1, gc2);
        TLOAD(loaded2, gc3);
        TADD(sum1, sum1, loaded1);
        TADD(sum2, sum2, loaded2);
    }
    scalar_tile mean_t, rstd_t, c2, c3;
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

template <typename dtype>
inline void dx_groups(dtype *dy, dtype *x, float *rstd, dtype *gamma,
                      float *workspace, dtype *dx, int64_t N, int64_t C, int64_t G,
                      int64_t D, int64_t n, int64_t g0, int64_t active_g) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using htile = Tile<Location::Vec, dtype, kPhysRows, kPhysCols, BLayout::CubeM32, -1, -1>;
    using ftile = Tile<Location::Vec, float, kPhysRows, kPhysCols, BLayout::CubeM32, -1, -1>;
    using vtile = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, -1, 1>;
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
template <typename dtype, typename gm_h, typename tile_h, typename tile_f>
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

// A1/A2 n-block path: one Tile holds batches [n0, n0+rows) of one group g as
// [rows, D] (D = Cols), so x/dY load with row stride C and mean/rstd/c2/c3
// with row stride G, and gamma is one [1, D] row broadcast over the batches.
// Chosen when it fills at least as many Tile rows as packing the groups of
// one batch.
inline bool use_nblock(int64_t N, int64_t G, int64_t D) {
    const int64_t nrows = N < kPhysRows ? N : kPhysRows;
    const int64_t grows = G < kPhysRows ? G : kPhysRows;
    return is_tile_cols(D) && nrows >= grows;
}

template <typename dtype, int Cols> struct NBlockTiles {
    using T = Tiles<dtype, Cols>;
    using RowH = Tile<Location::Vec, dtype, kPhysRows, Cols, BLayout::CubeM32, 1, Cols>;
    using RowF = Tile<Location::Vec, float, kPhysRows, Cols, BLayout::CubeM32, 1, Cols>;
};

// Stage A1 for batches [n0, n0+rows) of group g.
template <typename dtype, int Cols>
inline void fused_params_nblock(dtype *dy, dtype *x, float *mean, float *rstd,
                                dtype *gamma, float *workspace, int64_t N,
                                int64_t C, int64_t G, int64_t g, int64_t n0,
                                int64_t rows, float s) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using B = NBlockTiles<dtype, Cols>;
    using T = typename B::T;
    const int64_t ng = n0 * G + g;
    const int64_t offset = n0 * C + g * Cols;
    gm_h gx(x + offset, rows, C), gdy(dy + offset, rows, C);
    gm_h gg(gamma + g * Cols, 1, Cols);
    GFloat gm(mean + ng, rows, G), gr(rstd + ng, rows, G);
    GFloat gc2(workspace + ng, rows, G), gc3(workspace + N * G + ng, rows, G);
    typename T::H h(rows, Cols);
    typename T::F dyf(rows, Cols), xf(rows, Cols);
    typename T::W w1(rows), w2(rows);
    typename B::RowH gh;
    typename B::RowF gf;
    TLOAD(gh, gg);
    TCVT(gf, gh);
    TLOAD(h, gdy);
    TCVT(dyf, h);
    TCOLEXPANDMUL(dyf, dyf, gf);
    TLOAD(h, gx);
    TCVT(xf, h);
    TROWSUM(w2, dyf);
    TMUL(xf, xf, dyf);
    TROWSUM(w1, xf);
    c2_c3_rows<T>(w1, w2, gm, gr, gc2, gc3, rows, s);
}

// Stage A2 for batches [n0, n0+rows) of group g.
template <typename dtype, int Cols>
inline void dx_nblock(dtype *dy, dtype *x, float *rstd, dtype *gamma,
                      float *workspace, dtype *dx, int64_t N, int64_t C,
                      int64_t G, int64_t g, int64_t n0, int64_t rows) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using B = NBlockTiles<dtype, Cols>;
    using T = typename B::T;
    const int64_t ng = n0 * G + g;
    const int64_t offset = n0 * C + g * Cols;
    gm_h gx(x + offset, rows, C), gdy(dy + offset, rows, C);
    gm_h gout(dx + offset, rows, C), gg(gamma + g * Cols, 1, Cols);
    GFloat gr(rstd + ng, rows, G), gc2(workspace + ng, rows, G),
        gc3(workspace + N * G + ng, rows, G);
    typename T::H h(rows, Cols);
    typename T::F xf(rows, Cols), out(rows, Cols);
    typename T::V r(rows), c2(rows), c3(rows);
    typename B::RowH gh;
    typename B::RowF gf;
    TLOAD(gh, gg);
    TCVT(gf, gh);
    TLOAD(r, gr);
    TLOAD(c2, gc2);
    TLOAD(c3, gc3);
    TLOAD(h, gdy);
    TCVT(out, h);
    TCOLEXPANDMUL(out, out, gf);
    TROWEXPANDMUL(out, out, r);
    TLOAD(h, gx);
    TCVT(xf, h);
    TROWEXPANDMUL(xf, xf, c2);
    TADD(out, out, xf);
    TROWEXPANDADD(out, out, c3);
    TCVT(h, out);
    TSTORE(gout, h);
}

// Groups per (n, task) in stages A1 and A2. Both stages must use the same
// ownership: A2 reads the c2/c3 that A1 wrote on the same PE, with no
// inter-PE barrier. The A1 rows branch packs whole tile_g chunks up to 32.
inline int64_t owned_groups(int64_t D, int64_t tile_g) {
    return is_tile_cols(D) ? tile_g * (32 / tile_g) : tile_g;
}

// Stage B column width: a power of two W | D, W <= 64, shrunk until C/W gives
// every PE a block; 0 when D is odd.
inline int64_t gamma_beta_cols(int64_t C, int64_t D) {
    int64_t w = 64;
    while (w >= 2 && D % w != 0)
        w /= 2;
    while (w > 2 && C / w < 4)
        w /= 2;
    return w >= 2 ? w : 0;
}

template <typename dtype, int W> struct GammaBetaTiles {
    using H = Tile<Location::Vec, dtype, kPhysRows, W, BLayout::CubeM32, -1, W>;
    using F = Tile<Location::Vec, float, kPhysRows, W, BLayout::CubeM32, -1, W>;
    using V = Tile<Location::Vec, float, kPhysRows, 1, BLayout::CubeM32, -1, 1>;
    using RowF = Tile<Location::Vec, float, kPhysRows, W, BLayout::CubeM32, 1, W>;
    using RowH = Tile<Location::Vec, dtype, kPhysRows, W, BLayout::CubeM32, 1, W>;
};

// Batches [n0, n0+rows) of channels [c, c+W) in group g:
// b = dY, t = dY * (X - mean) * rstd.
template <typename dtype, int W>
__attribute__((always_inline)) inline void
gamma_beta_terms(dtype *dy, dtype *x, float *mean, float *rstd, int64_t C,
                 int64_t G, int64_t g, int64_t c, int64_t n0, int64_t rows,
                 typename GammaBetaTiles<dtype, W>::F &b,
                 typename GammaBetaTiles<dtype, W>::F &t) {
    using T = GammaBetaTiles<dtype, W>;
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    gm_h gdy(dy + n0 * C + c, static_cast<int>(rows), static_cast<int>(C));
    gm_h gx(x + n0 * C + c, static_cast<int>(rows), static_cast<int>(C));
    gm_f gm(mean + n0 * G + g, static_cast<int>(rows), static_cast<int>(G));
    gm_f gr(rstd + n0 * G + g, static_cast<int>(rows), static_cast<int>(G));
    typename T::H h(rows);
    typename T::V m(rows), r(rows);
    TLOAD(h, gdy);
    TCVT(b, h);
    TLOAD(h, gx);
    TCVT(t, h);
    TLOAD(m, gm);
    TLOAD(r, gr);
    TROWEXPANDSUB(t, t, m);
    TROWEXPANDMUL(t, t, r);
    TMUL(t, t, b);
}

template <typename dtype, int W>
__attribute__((always_inline)) inline void
gamma_beta_store(dtype *dgamma, dtype *dbeta, int64_t c,
                 typename GammaBetaTiles<dtype, W>::RowF &sb,
                 typename GammaBetaTiles<dtype, W>::RowF &sg) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    typename GammaBetaTiles<dtype, W>::RowH h;
    gm_h gb(dbeta + c, 1, W), gg(dgamma + c, 1, W);
    TCVT(h, sb);
    TSTORE(gb, h);
    TCVT(h, sg);
    TSTORE(gg, h);
}

// Stage B for channels [c, c+W): batches fill the 32 Tile rows, are summed
// with TADD per 32-batch block, then reduced once with TCOLSUM.
template <typename dtype, int W>
inline void gamma_beta_block(dtype *dy, dtype *x, float *mean, float *rstd,
                             dtype *dgamma, dtype *dbeta, int64_t N, int64_t C,
                             int64_t G, int64_t D, int64_t c) {
    using T = GammaBetaTiles<dtype, W>;
    const int64_t g = c / D;
    const int64_t full = N / kPhysRows * kPhysRows, tail = N - full;
    typename T::RowF sb, sg;
    if (full > 0) {
        typename T::F accb(kPhysRows), accg(kPhysRows), b(kPhysRows),
            t(kPhysRows);
        gamma_beta_terms<dtype, W>(dy, x, mean, rstd, C, G, g, c, 0, kPhysRows,
                                   accb, accg);
        for (int64_t n0 = kPhysRows; n0 < full; n0 += kPhysRows) {
            gamma_beta_terms<dtype, W>(dy, x, mean, rstd, C, G, g, c, n0,
                                       kPhysRows, b, t);
            TADD(accb, accb, b);
            TADD(accg, accg, t);
        }
        TCOLSUM(sb, accb);
        TCOLSUM(sg, accg);
    }
    if (tail > 0) {
        typename T::F b(tail), t(tail);
        typename T::RowF tb, tg;
        gamma_beta_terms<dtype, W>(dy, x, mean, rstd, C, G, g, c, full, tail, b,
                                   t);
        TCOLSUM(tb, b);
        TCOLSUM(tg, t);
        if (full == 0) {
            gamma_beta_store<dtype, W>(dgamma, dbeta, c, tb, tg);
            return;
        }
        TADD(sb, sb, tb);
        TADD(sg, sg, tg);
    }
    gamma_beta_store<dtype, W>(dgamma, dbeta, c, sb, sg);
}

// Fused A1+A2+B: one PE owns group g for every batch. No Tile lives across
// loop iterations: each 32-batch block stores its dgamma/dbeta column
// partials to the workspace and a final pass reduces them, since loop-carried
// Tiles are mis-indexed through zero-mask PseudoEmptyTile hand pads
// (LinxISA/llvm-project#130). The live set fits the 256 KiB tile pool for
// D <= kFusedMaxCols; the final reduce is one Tile, so N <= 32*32.
constexpr int kFusedMaxCols = 128;

inline bool use_fused(int64_t N, int64_t G, int64_t D, int peNum) {
    return use_nblock(N, G, D) && G >= peNum && D <= kFusedMaxCols &&
           N <= kPhysRows * kPhysRows;
}

// Batches [n0, n0+rows) of group g (block nb): writes dX and the dbeta/dgamma
// column partials sum_n dY and sum_n dY*(X-mean)*rstd, from one load of dY/X.
template <typename dtype, int Cols>
inline void fused_block(dtype *dy, dtype *x, float *mean, float *rstd,
                        dtype *gamma, dtype *dx, float *part_b, float *part_g,
                        int64_t C, int64_t G, int64_t g, int64_t nb,
                        int64_t n0, int64_t rows, float s) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using GB = GammaBetaTiles<dtype, Cols>;
    using W = Tile<Location::Vec, float, kPhysRows, Cols, BLayout::CubeM32, -1, 1>;
    using V = typename GB::V;
    const int64_t ng = n0 * G + g;
    const int64_t offset = n0 * C + g * Cols;
    gm_h gx(x + offset, rows, C), gdy(dy + offset, rows, C);
    gm_h gout(dx + offset, rows, C), gg(gamma + g * Cols, 1, Cols);
    GFloat gm(mean + ng, rows, G), gr(rstd + ng, rows, G);
    GFloat gpb(part_b + nb * Cols, 1, Cols), gpg(part_g + nb * Cols, 1, Cols);
    typename GB::H h(rows);
    typename GB::F b(rows), xf(rows), t(rows), gdyf(rows), p(rows);
    W w1(rows), w2(rows);
    V zero(rows), sum1(rows), sum2(rows), m(rows), r(rows), c2(rows), c3(rows);
    typename GB::RowH gh;
    typename GB::RowF gf, tb, tg;
    TLOAD(gh, gg);
    TCVT(gf, gh);
    TLOAD(h, gdy);
    TCVT(b, h);
    TLOAD(h, gx);
    TCVT(xf, h);
    TLOAD(m, gm);
    TLOAD(r, gr);
    TCOLSUM(tb, b);
    TROWEXPANDSUB(t, xf, m);
    TROWEXPANDMUL(t, t, r);
    TMUL(t, t, b);
    TCOLSUM(tg, t);
    TSTORE(gpb, tb);
    TSTORE(gpg, tg);
    // c2/c3 from sum1 = sum_d gamma*dY*X, sum2 = sum_d gamma*dY.
    TCOLEXPANDMUL(gdyf, b, gf);
    TROWSUM(w2, gdyf);
    TMUL(p, xf, gdyf);
    TROWSUM(w1, p);
    auto v1 = TREDUCEPREFIXVIEW<V>(w1);
    auto v2 = TREDUCEPREFIXVIEW<V>(w2);
    TEXPANDS(zero, 0.0f);
    TADD(sum1, zero, v1);
    TADD(sum2, zero, v2);
    c2_c3(c2, c3, sum1, sum2, m, r, s);
    // dX = rstd*gamma*dY + c2*X + c3.
    TROWEXPANDMUL(gdyf, gdyf, r);
    TROWEXPANDMUL(xf, xf, c2);
    TADD(gdyf, gdyf, xf);
    TROWEXPANDADD(gdyf, gdyf, c3);
    TCVT(h, gdyf);
    TSTORE(gout, h);
}

// Reduces the nblocks [1,D] partial rows of group g into dbeta/dgamma.
template <typename dtype, int Cols>
inline void fused_reduce(float *part_b, float *part_g, dtype *dgamma,
                         dtype *dbeta, int64_t g, int64_t nblocks) {
    using GB = GammaBetaTiles<dtype, Cols>;
    GFloat gpb(part_b, nblocks, Cols), gpg(part_g, nblocks, Cols);
    typename GB::F pb(nblocks), pg(nblocks);
    typename GB::RowF sb, sg;
    TLOAD(pb, gpb);
    TLOAD(pg, gpg);
    TCOLSUM(sb, pb);
    TCOLSUM(sg, pg);
    gamma_beta_store<dtype, Cols>(dgamma, dbeta, g * Cols, sb, sg);
}

// Stage B logical [active_g,active_d], physical [32,64]. Separate N reductions
// so dbeta and dgamma do not keep five FP32 tiles live together.
template <typename dtype>
inline void gamma_beta_groups(dtype *dy, dtype *x, float *mean, float *rstd,
                              dtype *dgamma, dtype *dbeta, int64_t N, int64_t C,
                              int64_t G, int64_t D, int64_t g0, int64_t d0,
                              int64_t active_g, int64_t active_d) {
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using ht = Tile<Location::Vec, dtype, kPhysRows, kPhysCols, BLayout::CubeM32, -1, -1>;
    using ft = Tile<Location::Vec, float, kPhysRows, kPhysCols, BLayout::CubeM32, -1, -1>;
    using vt = Tile<Location::Vec, float, kPhysRows, 1, BLayout::CubeM32, -1, 1>;
    ht h(active_g, active_d);
    ft acc(active_g, active_d), a(active_g, active_d), b(active_g, active_d);
    vt m(active_g), r(active_g);

    TEXPANDS(acc, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        const int64_t offset = n * C + g0 * D + d0;
        gm_h gdy(dy + offset, static_cast<int>(active_g), static_cast<int>(D));
        TLOAD(h, gdy);
        TCVT(a, h);
        TADD(acc, acc, a);
    }
    {
        gm_h gb(dbeta + g0 * D + d0, static_cast<int>(active_g), static_cast<int>(D));
        TCVT(h, acc);
        TSTORE(gb, h);
    }

    TEXPANDS(acc, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        const int64_t offset = n * C + g0 * D + d0;
        gm_h gx(x + offset, static_cast<int>(active_g), static_cast<int>(D));
        gm_h gdy(dy + offset, static_cast<int>(active_g), static_cast<int>(D));
        gm_f gm(mean + n * G + g0, static_cast<int>(active_g), 1);
        gm_f gr(rstd + n * G + g0, static_cast<int>(active_g), 1);
        TLOAD(h, gx);
        TCVT(a, h);
        TLOAD(r, gr);
        TROWEXPANDMUL(a, a, r);
        TLOAD(h, gdy);
        TCVT(b, h);
        TMUL(a, a, b);
        TLOAD(m, gm);
        TROWEXPANDMUL(b, b, m);
        TROWEXPANDMUL(b, b, r);
        TSUB(a, a, b);
        TADD(acc, acc, a);
    }
    {
        gm_h gg(dgamma + g0 * D + d0, static_cast<int>(active_g), static_cast<int>(D));
        TCVT(h, acc);
        TSTORE(gg, h);
    }
}

} // namespace gn_grad_1d

// One 4PE kernel; tiling = [N, C, G, tile_d, tile_g, gb_tile_d, gb_tile_g].
// Execute all three stages with identical PE ownership for parameters and dx.
template <typename dtype, int peNum>
__attribute__((noinline)) void
group_norm_grad_1d_dynamic(dtype *dy, dtype *x, float *mean, float *rstd,
                           dtype *gamma, const int64_t *tiling,
                           float *workspace, dtype *dx, dtype *dgamma,
                           dtype *dbeta) {
    // Fused path: one pass over dY/X per group computes everything.
    {
        const gn_grad_1d::Shape shape(tiling);
        const int64_t N = shape.N, C = shape.C, G = shape.G, D = shape.D;
        if (gn_grad_1d::use_fused(N, G, D, peNum)) {
            const uint32_t tid = get_thread_idx();
            const float s = 1.0f / static_cast<float>(D);
            const int64_t nblocks =
                (N + gn_grad_1d::kPhysRows - 1) / gn_grad_1d::kPhysRows;
            float *part_b = workspace, *part_g = workspace + nblocks * C;
            gn_grad_1d::with_cols(D, [=]<int Cols>() {
                if constexpr (Cols <= gn_grad_1d::kFusedMaxCols) {
                    for (int64_t g = tid; g < G; g += peNum)
                        for (int64_t nb = 0; nb < nblocks; ++nb) {
                            const int64_t n0 = nb * gn_grad_1d::kPhysRows;
                            const int64_t rows =
                                N - n0 < gn_grad_1d::kPhysRows
                                    ? N - n0 : gn_grad_1d::kPhysRows;
                            gn_grad_1d::fused_block<dtype, Cols>(
                                dy, x, mean, rstd, gamma, dx,
                                part_b + g * nblocks * Cols,
                                part_g + g * nblocks * Cols, C, G, g, nb, n0,
                                rows, s);
                        }
                    for (int64_t g = tid; g < G; g += peNum)
                        gn_grad_1d::fused_reduce<dtype, Cols>(
                            part_b + g * nblocks * Cols,
                            part_g + g * nblocks * Cols, dgamma, dbeta, g,
                            nblocks);
                }
            });
            return;
        }
    }
    // Parameters and dx retain identical (n, group-block) ownership.
    // gamma/beta read only immutable inputs, not another PE's workspace.
    // Stage 1: fused parameters.
    {
        // CubeM32 FP32: 32 physical rows x 64 columns = 8 KiB.
        constexpr int64_t tD = gn_grad_1d::data_columns<dtype>();

        const gn_grad_1d::Shape shape(tiling);
        const uint32_t tid = get_thread_idx();
        const int64_t N = shape.N, C = shape.C, G = shape.G, D = shape.D;
        const int64_t requested_d = tiling[3] > 0 ? tiling[3] : D;
        const int64_t tile_d = requested_d < tD ? requested_d : tD;

        using Types = gn_grad_1d::TileTypes<dtype, 1, tD>;
        using gm_h = typename Types::gm_h;
        using gm_f = typename Types::gm_f;
        using tile_h = typename Types::tile_h;
        using tile_f = typename Types::tile_f;
        using tile_v = typename Types::tile_v;

        const int64_t tile_g = tiling[4];
        const int64_t outer_g = (G + tile_g - 1) / tile_g;
        const float s = 1.0f / static_cast<float>(D);
        const int64_t fr = gn_grad_1d::fold_cols(D);
        if (gn_grad_1d::use_nblock(N, G, D)) {
            const int64_t nb = (N + gn_grad_1d::kPhysRows - 1) / gn_grad_1d::kPhysRows;
            gn_grad_1d::with_cols(D, [&]<int Cols>() {
                for (int64_t task = tid; task < G * nb; task += peNum) {
                    const int64_t g = task / nb;
                    const int64_t n0 = (task % nb) * gn_grad_1d::kPhysRows;
                    const int64_t rows = N - n0 < gn_grad_1d::kPhysRows
                                             ? N - n0 : gn_grad_1d::kPhysRows;
                    gn_grad_1d::fused_params_nblock<dtype, Cols>(
                        dy, x, mean, rstd, gamma, workspace, N, C, G, g, n0,
                        rows, s);
                }
            });
        } else if (gn_grad_1d::is_tile_cols(D)) {
            const int64_t own_g = gn_grad_1d::owned_groups(D, tile_g);
            const int64_t outer_r = (G + own_g - 1) / own_g;
            gn_grad_1d::with_cols(D, [&]<int Cols>() {
                for (int64_t task = tid; task < N * outer_r; task += peNum) {
                    const int64_t n = task / outer_r;
                    const int64_t g0 = (task % outer_r) * own_g;
                    const int64_t rows = G - g0 < own_g ? G - g0 : own_g;
                    gn_grad_1d::fused_params_rows<dtype, Cols>(
                        dy, x, mean, rstd, gamma, workspace, N, C, G, n, g0,
                        rows, s);
                }
            });
        } else if (fr > 0) {
            gn_grad_1d::with_cols(fr, [&]<int Cols>() {
                for (int64_t task = tid; task < N * outer_g; task += peNum) {
                    const int64_t n = task / outer_g;
                    const int64_t g0 = (task % outer_g) * tile_g;
                    const int64_t end_g = g0 + tile_g < G ? g0 + tile_g : G;
                    for (int64_t g = g0; g < end_g; ++g)
                        gn_grad_1d::fused_params_fold<dtype, Cols>(
                            dy, x, mean, rstd, gamma, workspace, N, C, G, n, g,
                            D / fr, s);
                }
            });
        } else {
            for (int64_t task = tid; task < N * outer_g; task += peNum) {
                const int64_t n = task / outer_g;
                const int64_t g0 = (task % outer_g) * tile_g;
                const int64_t end_g = g0 + tile_g < G ? g0 + tile_g : G;
                for (int64_t g = g0; g < end_g; ++g) {
                  gn_grad_1d::fused_params_group<dtype, gm_h, gm_f, tile_h,
                                                 tile_f, tile_v>(
                      dy, x, mean, rstd, gamma, workspace + n * G + g, N, C, G,
                      D, n, g, s, tile_d);
                }
            }
        }
    }
    // Stage 2: dx.
    {
        // CubeM32 FP32: 32 physical rows x 64 columns = 8 KiB.
        constexpr int64_t tD = gn_grad_1d::data_columns<dtype>();

        const gn_grad_1d::Shape shape(tiling);
        const uint32_t tid = get_thread_idx();
        const int64_t N = shape.N, C = shape.C, G = shape.G, D = shape.D;
        const int64_t requested_d = tiling[3] > 0 ? tiling[3] : D;
        const int64_t tile_d = requested_d < tD ? requested_d : tD;

        using Types = gn_grad_1d::TileTypes<dtype, 1, tD>;
        using gm_h = typename Types::gm_h;
        using gm_f = typename Types::gm_f;
        using tile_h = typename Types::tile_h;
        using tile_f = typename Types::tile_f;
        using tile_v = typename Types::tile_v;

        const int64_t tile_g = tiling[4];
        const int64_t own_g = gn_grad_1d::owned_groups(D, tile_g);
        const int64_t outer_own = (G + own_g - 1) / own_g;
        // Same (g, n-block) task order as Stage 1, so c2/c3 stay PE-local.
        if (gn_grad_1d::use_nblock(N, G, D)) {
            const int64_t nb = (N + gn_grad_1d::kPhysRows - 1) / gn_grad_1d::kPhysRows;
            gn_grad_1d::with_cols(D, [&]<int Cols>() {
                for (int64_t task = tid; task < G * nb; task += peNum) {
                    const int64_t g = task / nb;
                    const int64_t n0 = (task % nb) * gn_grad_1d::kPhysRows;
                    const int64_t rows = N - n0 < gn_grad_1d::kPhysRows
                                             ? N - n0 : gn_grad_1d::kPhysRows;
                    gn_grad_1d::dx_nblock<dtype, Cols>(dy, x, rstd, gamma,
                                                       workspace, dx, N, C, G,
                                                       g, n0, rows);
                }
            });
        } else {
            for (int64_t task = tid; task < N * outer_own; task += peNum) {
                const int64_t n = task / outer_own;
                const int64_t g_begin = (task % outer_own) * own_g;
                const int64_t g_end = g_begin + own_g < G ? g_begin + own_g : G;
                for (int64_t g0 = g_begin; g0 < g_end; g0 += tile_g) {
                  const int64_t active_g = g_end - g0 < tile_g ? g_end - g0 : tile_g;
                  if (D <= gn_grad_1d::kPhysCols && tile_g > 1) {
                    gn_grad_1d::dx_groups(dy, x, rstd, gamma, workspace, dx, N,
                                          C, G, D, n, g0, active_g);
                  } else {
                    gn_grad_1d::dx_group<dtype, gm_h, gm_f, tile_h, tile_f,
                                         tile_v>(
                        dy, x, rstd, gamma, workspace + n * G + g0, dx, N, C, G,
                        D, n, g0, tile_d);
                  }
                }
            }
        }
    }
    // Stage 3: gamma/beta.
    {
        // CubeM32 FP32: 32 physical rows x 64 columns = 8 KiB.
        constexpr int64_t tD = gn_grad_1d::data_columns<dtype>();

        const gn_grad_1d::Shape shape(tiling);
        const uint32_t tid = get_thread_idx();
        const int64_t N = shape.N, C = shape.C, G = shape.G, D = shape.D;
        const int64_t requested_d = tiling[5] > 0 ? tiling[5] : D;
        const int64_t tile_d = requested_d < tD ? requested_d : tD;

        using Types = gn_grad_1d::TileTypes<dtype, 1, tD>;
        using gm_h = typename Types::gm_h;
        using gm_f = typename Types::gm_f;
        using tile_h = typename Types::tile_h;
        using tile_f = typename Types::tile_f;
        using tile_v = typename Types::tile_v;

        const int64_t tile_g = tiling[6];
        if (const int64_t w = gn_grad_1d::gamma_beta_cols(C, D); w > 0) {
            gn_grad_1d::with_cols(w, [&]<int W>() {
                for (int64_t blk = tid; blk < C / W; blk += peNum)
                    gn_grad_1d::gamma_beta_block<dtype, W>(
                        dy, x, mean, rstd, dgamma, dbeta, N, C, G, D, blk * W);
            });
            return;
        }
        if (tile_d <= gn_grad_1d::kPhysCols) {
            const int64_t outer_g = (G + tile_g - 1) / tile_g;
            const int64_t outer_d = (D + tile_d - 1) / tile_d;
            // Each PE owns complete output blocks; N is reduced locally.
            for (int64_t task = tid; task < outer_g * outer_d; task += peNum) {
              const int64_t g0 = (task / outer_d) * tile_g;
              const int64_t d0 = (task % outer_d) * tile_d;
              const int64_t vg = G - g0 < tile_g ? G - g0 : tile_g;
              const int64_t vd = D - d0 < tile_d ? D - d0 : tile_d;
              gn_grad_1d::gamma_beta_groups(dy, x, mean, rstd, dgamma, dbeta, N,
                                            C, G, D, g0, d0, vg, vd);
            }
            return;
        }
        for (int64_t g = tid; g < G; g += peNum) {
            gn_grad_1d::dbeta_group<dtype, gm_h, tile_h, tile_f>(
                dy, dbeta, N, C, D, tile_d, g);
            gn_grad_1d::dgamma_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
                dy, x, mean, rstd, dgamma, N, C, G, D, tile_d, g);
        }
    }
}

#endif // SUPERNPU_GROUP_NORM_GRAD_1D_PTO_HPP
