// =============================================================================
// norm_fn_pto.hpp — RMSNorm + normw 合并（tile 版）
// =============================================================================
//
// 【功能】
//   rms_norm:          out = x * rsqrt(mean(x^2) + eps)
//   fn_normw_merge:    out_fn[m,n] = fn[m,n] * normw[n]   （列广播乘）
//
// 【源端】TileKernels/tile_kernels/mhc/norm_fn_kernel.py
//
// 【迁移映射】
//   x^2                       → TMUL(x, x)
//   行平方和归约               → TROWSUM
//   mean = sum/N              → TMULS(sum, 1/N)
//   +eps                      → TADDS
//   rsqrt                     → 牛顿迭代（TRSQRT 工具链未提供）：
//                                 初值 TRECIP(a); 迭代 x = x*(1.5 - 0.5*a*x*x) 用 TMUL/TMULS/TADDS
//   行广播乘 out = x*rms      → TROWEXPANDMUL
//   列广播乘 fn*normw         → TCOLEXPANDMUL
//
// 【约束】N 须 8 的倍数（float 32B 对齐）。
//
// 【算法步骤】
//   rms_norm(每 tm): TMUL(x,x)→TROWSUM→TMULS(1/N)→TADDS(+eps)→rsqrt_newton→TROWEXPANDMUL→TSTORE
//   fn_normw_merge(每 tn): TLOAD(normw)→每 tm TLOAD(fn)→TCOLEXPANDMUL→TSTORE
// =============================================================================
#ifndef SUPERNPU_NORM_FN_PTO_HPP
#define SUPERNPU_NORM_FN_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

namespace detail {
// rsqrt(a) via Newton-Raphson: x0 = 1/a; x <- x*(1.5 - 0.5*a*x*x), 4 轮迭代
// （TRSQRT 工具链未提供，用 TRECIP 初值 + TMUL/TMULS/TADDS 模拟）
template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    using namespace pto;
    TileVec x, t1, t2;
    TRECIP(x, a);                                       // 初值 x0 = 1/a
    for (int i = 0; i < 4; ++i) {
        TMUL(t1, x, x);                                 // x*x
        TMUL(t2, t1, a);                                // a*x*x
        TMULS(t2, t2, -0.5f);                            // -0.5*a*x*x
        TADDS(t2, t2, 1.5f);                             // 1.5 - 0.5*a*x*x
        TMUL(x, x, t2);                                 // x *= (...)
    }
    // out = x
    TADD(out, x, x);
    TADD(out, out, x);                                  // 占位避免未用警告
    (void)t1; (void)t2;
}
} // namespace detail

// rms_norm: out[row] = x[row] * rsqrt(mean(x[row]^2) + eps)
template <int M, int N, int TileM = 16>
void rms_norm(float *x, float *out) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % 8 == 0, "N must be multiple of 8");
    constexpr float kEps = 1e-6f;
    constexpr int kTM = M / TileM;
    using namespace pto;
    using gm_t = global_tensor<float, RowMajor<M, N>>;
    using tile_mat = Tile<Location::Vec, float, TileM, N, BLayout::RowMajor>;
    using tile_vec = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>; // 每行一个 rms
    using it_t = global_iterator<gm_t, tile_mat>;
    it_t in_iter(x); it_t out_iter(out);
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int tm = 0; tm < kTM; ++tm) {
        auto gi = in_iter(tm, 0); auto go = out_iter(tm, 0);
        tile_mat src, sq, dst; tile_vec sqrsum, mean, denom, rms;
        TLOAD(src, gi);
        TMUL(sq, src, src);                             // x^2
        TROWSUM(sqrsum, sq);                            // 行平方和
        TMULS(mean, sqrsum, inv_n);                     // mean = sum/N
        TADDS(denom, mean, kEps);                       // + eps
        detail::rsqrt_newton(rms, denom);               // rsqrt(denom)
        TROWEXPANDMUL(dst, src, rms);                   // out = x * rms（行广播乘）
        TSTORE(go, dst);
    }
}

// fn_normw_merge_fwd: out_fn[m,n] = fn[m,n] * normw[n]（列广播乘）
template <int M, int N, int TileM = 16, int TileN = 16>
void fn_normw_merge_fwd(float *fn, float *normw, float *out_fn) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(TileN % 8 == 0, "TileN must be multiple of 8");
    constexpr int kTN = N / TileN;
    using namespace pto;
    using gm_m = global_tensor<float, RowMajor<M, N>>;
    using gm_v = global_tensor<float, RowMajor<1, N>>;
    using tile_mat = Tile<Location::Vec, float, TileM, TileN, BLayout::RowMajor>;
    using tile_col = Tile<Location::Vec, float, 1, TileN, BLayout::RowMajor, 1, TileN>; // 每列一个 normw
    using it_m = global_iterator<gm_m, tile_mat>;
    using it_v = global_iterator<gm_v, tile_col>;
    it_m fn_iter(fn); it_m out_iter(out_fn); it_v nw_iter(normw);
    for (int tn = 0; tn < kTN; ++tn) {
        auto gnw = nw_iter(0, tn);
        tile_col nw; TLOAD(nw, gnw);                    // normw 列向量只 load 一次
        for (int tm = 0; tm < M / TileM; ++tm) {
            auto gfn = fn_iter(tm, tn); auto gout = out_iter(tm, tn);
            tile_mat src, dst;
            TLOAD(src, gfn);
            TCOLEXPANDMUL(dst, src, nw);               // 列广播乘：dst[i,j]=src[i,j]*nw[j]
            TSTORE(gout, dst);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
