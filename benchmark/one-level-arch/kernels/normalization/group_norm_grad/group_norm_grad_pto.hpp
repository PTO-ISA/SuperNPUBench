// =============================================================================
// group_norm_grad_pto.hpp — GroupNorm backward, HxW > 1 (one-level PTO)
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
// tiling[5] = {N, C, G, HxW, tile_hw}
//   tile_hw <= 0 → min(HxW, tCap); spatial R-split when HxW > tile_hw.
//   dX still requires HxW <= tCap (one spatial tile; RF-limited).
//
// tCap: logical tile >= 512B (TileOP IsValidActiveSize / TSize=1..7).
//   fp16 → Cols>=256; fp32 data tiles → Cols>=128. Reduction/broadcast
//   tile_v follows the hardware contract and uses physical Columns=1.
// Cols=1024 like rms_norm overflows Tile RF here.
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


inline int64_t workspace_elems(int64_t N, int64_t C, int64_t G) {
    return 2 * N * C + 2 * N * G;
}

constexpr int kMaxPeCount = 64;
static volatile uint32_t kPeBarrier[kMaxPeCount] = {};

__attribute__((noinline)) inline uint32_t read_pe_id() {
    return get_thread_idx();
}

template <int peNum>
__attribute__((noinline)) void pe_barrier(uint32_t phase) {
    static_assert(peNum > 0 && peNum <= kMaxPeCount);
    if constexpr (peNum > 1) {
        const uint32_t pe = read_pe_id();
        kPeBarrier[pe] = phase;
        for (int participant = 0; participant < peNum; ++participant) {
            while (kPeBarrier[participant] < phase) {
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 1: spatial reduce for one (n, c) → ds[nc], db[nc]  (HxW R-split)
//
// Torch: ComputeInternalGradientsCUDAKernel
//   grid  = N * C          // 每个 (n,c) 一个 block；本函数 = 其中一个 block
//   block = (HxW < 512) ? 32 : 512
//   线程: threadIdx.x 沿 hw 做 grid-stride + warp/block reduce
//   → ds[n,c]=Σ dY*X, db[n,c]=Σ dY
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void spatial_reduce_nc(dtype *dy, dtype *x, float *ds, float *db,
                              int64_t N, int64_t C, int64_t HxW,
                              int64_t tile_hw, int64_t n, int64_t c) {
    const int64_t nc = n * C + c;
    const int64_t base = nc * HxW;

    gm_f gds(ds + nc, static_cast<int>(N * C), 1);
    gm_f gdb(db + nc, static_cast<int>(N * C), 1);

    tile_v ds_acc(1);
    tile_v db_acc(1);
    tile_v cur(1);
    TEXPANDS(ds_acc, 0.0f);
    TEXPANDS(db_acc, 0.0f);

    // Torch: for (hw = threadIdx.x; hw < HxW; hw += blockDim.x) + BlockReduce
    // PTO: tile 覆盖一段 HxW，TROWSUM 代替 block 内线程归约
    for (int64_t hw0 = 0; hw0 < HxW; hw0 += tile_hw) {
        const size_t vh = static_cast<size_t>(
            (hw0 + tile_hw <= HxW) ? tile_hw : (HxW - hw0));
        const int64_t offset = base + hw0;

        gm_h gdy(dy + offset, static_cast<int>(N * C), static_cast<int>(HxW));
        gm_h gx(x + offset, static_cast<int>(N * C), static_cast<int>(HxW));

        tile_h h0(1, vh);
        tile_f x_f(1, vh);
        tile_f dy_f(1, vh);
        tile_f prod(1, vh);

        TLOAD(h0, gx);
        TCVT(x_f, h0);
        TLOAD(h0, gdy);
        TCVT(dy_f, h0);
        TMUL(prod, dy_f, x_f);
        TROWSUM(cur, prod);
        TADD(ds_acc, ds_acc, cur);
        TROWSUM(cur, dy_f);
        TADD(db_acc, db_acc, cur);
    }

    TSTORE(gds, ds_acc);
    TSTORE(gdb, db_acc);
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
    const size_t active_d = static_cast<size_t>(D);

    gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
    gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);
    gm_f gds(ds + n * C + c0, 1, static_cast<int>(C));
    gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
    gm_f gc2(c2_buf + ng, static_cast<int>(N * G), 1);
    gm_f gc3(c3_buf + ng, static_cast<int>(N * G), 1);

    tile_f ds_f(1, active_d);
    tile_f db_f(1, active_d);
    tile_f gamma_f(1, active_d);
    tile_f t0(1, active_d);
    tile_h h0(1, active_d);
    tile_v mean_t(1);
    tile_v rstd_t(1);
    tile_v sum1(1);
    tile_v sum2(1);
    tile_v c2(1);
    tile_v c3(1);

    TLOAD(ds_f, gds);
    TLOAD(db_f, gdb);
    TLOAD(mean_t, gmean);
    TLOAD(rstd_t, grstd);

    {
        gm_h gg(gamma + c0, 1, static_cast<int>(C));
        TLOAD(h0, gg);
        TCVT(gamma_f, h0);
    }

    // Torch: threads 各算 ds*gamma / db*gamma 再 reduce → sum1/sum2
    TMUL(t0, ds_f, gamma_f);
    TROWSUM(sum1, t0);
    TMUL(t0, db_f, gamma_f);
    TROWSUM(sum2, t0);

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
    // TCVT must keep matching logical shapes (PTO 0.58). tile_h Cols=tCap
    // (fp16→256) while tile_v Cols=128, so convert via tile_f then TROWSUM.
    {
        gm_h gg(gamma + c, 1, 1);
        tile_h hg(1, 1);
        tile_f gf(1, 1);
        tile_v gv(1);
        TLOAD(hg, gg);
        TCVT(gf, hg);
        TROWSUM(gv, gf);
        TMUL(c1, gv, rstd_t);
    }

    const int64_t base = (n * C + c) * HxW;
    for (int64_t hw0 = 0; hw0 < HxW; hw0 += tile_hw) {
        const size_t active_hw = static_cast<size_t>(
            (hw0 + tile_hw <= HxW) ? tile_hw : (HxW - hw0));
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

// ---------------------------------------------------------------------------
// Step 4a: dbeta  — dbeta[c] = Σ_n db[n,c]
//
// Torch: GammaBetaBackwardCUDAKernel1/2（与 dgamma 同一次 launch）
//   N<=128: grid=ceil(C/256), block=256；每线程一个 c，循环 n
//   N>128:  grid=ceil(C/32),  block=dim3(32,16)
// 本函数按 group 一次写 D 个通道（对 N 串行累加）
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f>
inline void dbeta_group(float *db, dtype *dbeta, int64_t N, int64_t C,
                        int64_t D, int64_t g) {
    const int64_t c0 = g * D;
    const size_t active_d = static_cast<size_t>(D);

    tile_f acc(1, active_d);
    tile_f cur(1, active_d);
    tile_h h0(1, active_d);
    TEXPANDS(acc, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
        TLOAD(cur, gdb);
        TADD(acc, acc, cur);
    }

    gm_h gout(dbeta + c0, 1, static_cast<int>(C));
    TCVT(h0, acc);
    TSTORE(gout, h0);
}

// ---------------------------------------------------------------------------
// Step 4b: dgamma — dgamma[c] = Σ_n (ds - db*mean)*rstd
//
// Torch: 与 dbeta 同 Kernel1/2 launch（见上）
// ---------------------------------------------------------------------------
template <typename dtype, typename gm_h, typename gm_f, typename tile_h,
          typename tile_f, typename tile_v>
inline void dgamma_group(float *ds, float *db, float *mean, float *rstd,
                         dtype *dgamma, int64_t N, int64_t C, int64_t G,
                         int64_t D, int64_t g) {
    const int64_t c0 = g * D;
    const size_t active_d = static_cast<size_t>(D);

    tile_f acc(1, active_d);
    tile_f ds_f(1, active_d);
    tile_f db_f(1, active_d);
    tile_f t0(1, active_d);
    tile_h h0(1, active_d);
    tile_v mean_t(1);
    tile_v rstd_t(1);
    TEXPANDS(acc, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t ng = n * G + g;
        gm_f gds(ds + n * C + c0, 1, static_cast<int>(C));
        gm_f gdb(db + n * C + c0, 1, static_cast<int>(C));
        gm_f gmean(mean + ng, static_cast<int>(N * G), 1);
        gm_f grstd(rstd + ng, static_cast<int>(N * G), 1);

        TLOAD(ds_f, gds);
        TLOAD(db_f, gdb);
        TLOAD(mean_t, gmean);
        TLOAD(rstd_t, grstd);

        TROWEXPANDMUL(t0, db_f, mean_t);
        TSUB(t0, ds_f, t0);
        TROWEXPANDMUL(t0, t0, rstd_t);
        TADD(acc, acc, t0);
    }

    gm_h gout(dgamma + c0, 1, static_cast<int>(C));
    TCVT(h0, acc);
    TSTORE(gout, h0);
}

} // namespace gn_grad

// tiling: [N, C, G, HxW, tile_hw]
// workspace: float[2*N*C + 2*N*G]
//
// 入口循环 ↔ Torch 各 kernel 的 grid 遍历：
//   for n,c spatial_reduce  ↔ grid = N*C
//   for n,g fused_params    ↔ grid = dim3(N,G)
//   for n,c dx_nc           ↔ numel 上 gpu_kernel 线性网格
//   for g  dbeta/dgamma     ↔ 按通道写回（Kernel1/2）
template <typename dtype, int peNum>
void group_norm_grad(dtype *dy, dtype *x, float *mean, float *rstd,
                     dtype *gamma, const int64_t *tiling, dtype *dx,
                     dtype *dgamma, dtype *dbeta, float *workspace) {
    static_assert(peNum == 4, "normalization kernels support only 4PE");
    // One 512-element spatial tile matches the source algorithm's maximum block.
    // Larger HxW values are handled by the tile_hw loops.
    constexpr int64_t tCap = 512;
    constexpr int64_t tV = 1; // row-reduction/broadcast physical Columns=1

    const int64_t N = tiling[0];
    const int64_t C = tiling[1];
    const int64_t G = tiling[2];
    const int64_t HxW = tiling[3];
    const int64_t tile_hw =
        tiling[4] > 0 ? tiling[4] : (HxW < tCap ? HxW : tCap);
    const uint32_t tid = gn_grad::read_pe_id();
    if (N <= 0 || C <= 0 || G <= 0 || HxW <= 0 || C % G != 0) {
        return;
    }
    const int64_t D = C / G;
    if (tile_hw <= 0 || tile_hw > tCap || D > tCap ||
        tid >= static_cast<uint32_t>(peNum)) {
        return;
    }

    float *ds = workspace;
    float *db = workspace + N * C;
    float *c2_buf = workspace + 2 * N * C;
    float *c3_buf = c2_buf + N * G;
    using gm_h = global_tensor<dtype, RowMajor<-1, -1>>;
    using gm_f = global_tensor<float, RowMajor<-1, -1>>;
    using tile_h =
        Tile<Location::Vec, dtype, 1, tCap, BLayout::RowMajor, -1, -1>;
    using tile_f =
        Tile<Location::Vec, float, 1, tCap, BLayout::RowMajor, -1, -1>;
    using tile_v =
        Tile<Location::Vec, float, 1, tV, BLayout::RowMajor, -1, 1>;

    const float s = 1.0f / static_cast<float>(D * HxW);

    for (int64_t nc = tid; nc < N * C; nc += peNum) {
        const int64_t n = nc / C;
        const int64_t c = nc % C;
        gn_grad::spatial_reduce_nc<dtype, gm_h, gm_f, tile_h, tile_f,
                                   tile_v>(dy, x, ds, db, N, C, HxW,
                                           tile_hw, n, c);
    }
    gn_grad::pe_barrier<peNum>(1);

    for (int64_t ng = tid; ng < N * G; ng += peNum) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        gn_grad::fused_params_group<dtype, gm_h, gm_f, tile_h, tile_f,
                                    tile_v>(gamma, mean, rstd, ds, db,
                                            c2_buf, c3_buf, N, C, G, D, n, g,
                                            s);
    }
    gn_grad::pe_barrier<peNum>(2);

    for (int64_t nc = tid; nc < N * C; nc += peNum) {
        const int64_t n = nc / C;
        const int64_t c = nc % C;
        gn_grad::dx_nc<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            dy, x, gamma, rstd, c2_buf, c3_buf, dx, N, C, G, D, HxW,
            tile_hw, n, c);
    }

    for (int64_t g = tid; g < G; g += peNum) {
        gn_grad::dbeta_group<dtype, gm_h, gm_f, tile_h, tile_f>(db, dbeta, N,
                                                                C, D, g);
        gn_grad::dgamma_group<dtype, gm_h, gm_f, tile_h, tile_f, tile_v>(
            ds, db, mean, rstd, dgamma, N, C, G, D, g);
    }
    gn_grad::pe_barrier<peNum>(3);
}

#endif // SUPERNPU_GROUP_NORM_GRAD_PTO_HPP
