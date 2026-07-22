// =============================================================================
// sinkhorn_pto.hpp — Sinkhorn 行列交替归一化（前向）（tile 版）
// =============================================================================
//
// 【功能】
//   对 (N,H,H) 的每个 H×H 矩阵做 Sinkhorn 归一化使矩阵双随机近似：
//     1) 行方向 softmax：x = exp(x - row_max); x = x / (row_sum + eps); x += eps
//     2) 列归一化：x = x / (col_sum + eps)
//     3) repeat 次行列交替归一化
//
// 【源端】TileKernels/tile_kernels/mhc/sinkhorn_kernel.py (_mhc_sinkhorn_fwd)
//
// 【迁移映射】Sinkhorn 本质是"行/列归约 + 行/列广播运算"反复组合：
//   行 max                → TROWMAX
//   行减（数值稳定 softmax）→ TROWEXPANDSUB(x - row_max)
//   exp                   → TEXP
//   行求和                 → TROWSUM
//   +eps                  → TADDS
//   行广播除 → 1/sum+行广播乘 → TRECIP + TROWEXPANDMUL（TROWEXPANDDIV 工具链未提供）
//   列求和                 → TCOLSUM
//   列广播乘               → TCOLEXPANDMUL
//
// 【约束】
//   - global_iterator 仅 2 参 → 3D(N,H,H) 展平为 2D(N*H, H)
//   - TileM 须 8 的倍数（float 32B 对齐）
//   - eps 用 constexpr 局部量（float 非类型模板参数工具链不支持）
//
// 【算法步骤（每 (n,ti,tj) tile）】
//   行 softmax: TROWMAX→TROWEXPANDSUB→TEXP→TROWSUM→TADDS(+eps)→TRECIP→TROWEXPANDMUL→TADDS(+eps)
//   列归一化:   TCOLSUM→TADDS(+eps)→TRECIP→TCOLEXPANDMUL
//   repeat-1 次重复 行/列 归一化
// =============================================================================
#ifndef SUPERNPU_SINKHORN_PTO_HPP
#define SUPERNPU_SINKHORN_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int Hidden, int Repeat = 1, int TileM = 16>
void sinkhorn_fwd(float *comb_res_mix, float *comb_res_mix_out) {
    static_assert(NumTokens > 0 && Hidden > 0 && Repeat >= 1, "dim must be positive");
    static_assert(TileM % 8 == 0, "TileM must be multiple of 8");
    constexpr float kEps = 1e-20f;
    constexpr int kTiles = Hidden / TileM;
    using namespace pto;
    using gm_t = global_tensor<float, RowMajor<NumTokens * Hidden, Hidden>>; // 3D 展平 2D
    using tile_mat = Tile<Location::Vec, float, TileM, TileM, BLayout::RowMajor>;
    using tile_vec = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>; // 行向量(每行一个)
    using tile_cvec = Tile<Location::Vec, float, 1, TileM, BLayout::RowMajor, 1, TileM>; // 列向量(每列一个)
    using it_t = global_iterator<gm_t, tile_mat>;
    it_t in_iter(comb_res_mix); it_t out_iter(comb_res_mix_out);

    for (int n = 0; n < NumTokens; ++n) {
        for (int ti = 0; ti < kTiles; ++ti) {
            for (int tj = 0; tj < kTiles; ++tj) {
                auto gin = in_iter(n * kTiles + ti, tj); auto gout = out_iter(n * kTiles + ti, tj);
                tile_mat x, t; tile_vec rmax, rsum, denom, recip; tile_cvec csum, cdenom, crecip;
                TLOAD(x, gin);
                // ---- 行方向 softmax: x = exp(x-row_max); x /= (row_sum+eps); x += eps ----
                TROWMAX(rmax, x);                         // 行最大值（数值稳定）
                TROWEXPANDSUB(t, x, rmax);                // t = x - row_max（行广播减）
                TEXP(x, t);                               // x = exp(t)
                TROWSUM(rsum, x);                         // 行求和
                TADDS(denom, rsum, kEps);                 // denom = row_sum + eps
                TRECIP(recip, denom);                     // 1/denom（替代行除）
                TROWEXPANDMUL(t, x, recip);               // t = x * (1/denom) = x/denom
                TADDS(x, t, kEps);                        // x += eps
                // ---- 列归一化: x /= (col_sum+eps) ----
                TCOLSUM(csum, x);                         // 列求和
                TADDS(cdenom, csum, kEps);
                TRECIP(crecip, cdenom);
                TCOLEXPANDMUL(x, x, crecip);             // x = x * (1/col_denom)（列广播乘）
                // ---- repeat-1 次行列交替归一化 ----
                for (int r = 1; r < Repeat; ++r) {
                    TROWSUM(rsum, x);
                    TADDS(denom, rsum, kEps);
                    TRECIP(recip, denom);
                    TROWEXPANDMUL(t, x, recip);
                    TCOLSUM(csum, t);
                    TADDS(cdenom, csum, kEps);
                    TRECIP(crecip, cdenom);
                    TCOLEXPANDMUL(x, t, crecip);
                }
                TSTORE(gout, x);
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
