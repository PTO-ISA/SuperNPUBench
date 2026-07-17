// =============================================================================
// normalize_weight_pto.hpp — MoE top-k 路由权重的行归一化（PTO 一层 tile 版）
// =============================================================================
//
// 【功能】
//   把每个 token 的 top-k 路由权重归一化使其和为 1：
//     denominator[row]      = sum(topk_weights[row, :]) + eps
//     normalized[row, k]    = topk_weights[row, k] / denominator[row]
//   eps=1e-20 防止零除。输出两份：denominator (NumTokens×1) 与 normalized (NumTokens×NumTopk)。
//
// 【源端】TileKernels/tile_kernels/moe/normalize_weight_kernel.py
//   源端用 128 threads 每 thread 处理一行，T.unroll(num_topk) 串行求和 + T.vectorized 并行除。
//
// 【迁移映射】TileLang 原语 → PTO TileOP 指令（本工具链实际暴露的子集）
//   T.reduce_sum(行求和)            → TROWSUM(dst, src)  一指令归约整行
//   防 0 除 +eps                     → TADDS(dst, src, eps)  tile-标量加
//   行除（每元素 / 行和）            → TRECIP + TROWEXPANDMUL
//                                     （TROWEXPANDDIV 工具链未提供，故用 1/sum 后行广播乘替代）
//   T.copy(GM↔tile)                → TLOAD / TSTORE（形参为左值引用，须先 auto ref=iter()）
//   归约输出 1 宽 tile 违 32B 对齐   → tile_sum 物理取 8 宽、ValidCol=1（仿 reducesum_rowvec）
//
// 【约束】
//   - NumTopk 须为 8 的倍数（float：列宽 32B 对齐）。
//   - TileM 使 tile_mat、tile_sum 总大小 ≥ 128B（TLOAD/TSTORE 最小 128B，默认 TileM=16→512B）。
//
// 【算法步骤（每 tile）】
//   1) TLOAD 从 GM 取一个 (TileM × NumTopk) 权重 tile
//   2) TROWSUM 沿列归约得每行和 (TileM × 8 valid1)
//   3) TADDS 加 eps 得 denominator
//   4) TRECIP 求 1/denominator（替代除法）
//   5) TROWEXPANDMUL 把 1/denominator 行广播乘到每个权重 → normalized
//   6) TSTORE 写回 denominator 与 normalized
//   尾行（NumTokens 非 TileM 整数倍）用 ValidRow=kTailM 的 partial tile 处理。
// =============================================================================
#ifndef SUPERNPU_NORMALIZE_WEIGHT_PTO_HPP
#define SUPERNPU_NORMALIZE_WEIGHT_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int NumTopk, int TileM = 16>
void normalize_weight(float *topk_weights, float *denominator,
                      float *normalized_weights) {
    static_assert(NumTokens > 0 && NumTopk > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0, "NumTopk must be multiple of 8 (float 32B align)");
    constexpr float kEps = 1e-20f;                       // 防 0 除小常数
    constexpr int kTiles = NumTokens / TileM;            // 完整 tile 数
    constexpr int kTailM = NumTokens % TileM;            // 尾行数

    using namespace pto;
    // 全局张量：token × topk 行主序
    using gm_in  = global_tensor<float, RowMajor<NumTokens, NumTopk>>;
    using gm_den = global_tensor<float, RowMajor<NumTokens, 1>>;
    using gm_out = global_tensor<float, RowMajor<NumTokens, NumTopk>>;
    // tile 形状：mat=TileM×NumTopk 权重块；sum=TileM×8 valid1（每行一个行和，物理 8 宽满足 32B 对齐）
    using tile_mat = Tile<Location::Vec, float, TileM, NumTopk, BLayout::RowMajor>;
    using tile_sum = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;
    // 迭代器：从裸指针构造，operator(i,j) 取 (i,j) 块的 GM 引用
    using it_mat = global_iterator<gm_in,  tile_mat>;
    using it_den = global_iterator<gm_den, tile_sum>;
    using it_out = global_iterator<gm_out, tile_mat>;

    it_mat in_iter(topk_weights);                         // 输入迭代器
    it_out out_iter(normalized_weights);                  // 归一化输出迭代器
    it_den den_iter(denominator);                         // denominator 输出迭代器

    // 单个完整 tile 的处理（抽出为 lambda 供尾行复用逻辑）
    auto run = [&](auto &in_it, auto &out_it, auto &den_it, int t) {
        auto gi = in_it(t, 0);                            // GM 引用须先绑定左值（TLOAD 形参为引用）
        auto go = out_it(t, 0);
        auto gd = den_it(t, 0);
        tile_mat src, dst;
        tile_sum sum, denom, recip;
        TLOAD(src, gi);                                  // 1) GM -> tile 权重
        TROWSUM(sum, src);                              // 2) 每行求和 sum[row]=Σ_k src[row,k]
        TADDS(denom, sum, kEps);                        // 3) denom = sum + eps
        TRECIP(recip, denom);                           // 4) recip = 1/denom（替代行除）
        TROWEXPANDMUL(dst, src, recip);                 // 5) dst[row,k]=src[row,k]*recip[row]
        TSTORE(gd, denom);                              // 6a) 写 denominator
        TSTORE(go, dst);                                // 6b) 写 normalized
    };

    for (int t = 0; t < kTiles; ++t) run(in_iter, out_iter, den_iter, t);

    // 尾行：用 ValidRow=kTailM 的 partial tile（物理形状不变，仅有效行缩小）
    if constexpr (kTailM != 0) {
        using tile_mat_r = Tile<Location::Vec, float, TileM, NumTopk, BLayout::RowMajor, kTailM, NumTopk>;
        using tile_sum_r = Tile<Location::Vec, float, TileM, 8, BLayout::RowMajor, kTailM, 1>;
        using it_mat_r = global_iterator<gm_in,  tile_mat_r>;
        using it_out_r = global_iterator<gm_out, tile_mat_r>;
        using it_den_r = global_iterator<gm_den, tile_sum_r>;
        it_mat_r in_r(topk_weights);
        it_out_r out_r(normalized_weights);
        it_den_r den_r(denominator);
        auto gi = in_r(kTiles, 0);
        auto go = out_r(kTiles, 0);
        auto gd = den_r(kTiles, 0);
        tile_mat_r src, dst;
        tile_sum_r sum, denom, recip;
        TLOAD(src, gi);
        TROWSUM(sum, src);
        TADDS(denom, sum, kEps);
        TRECIP(recip, denom);
        TROWEXPANDMUL(dst, src, recip);
        TSTORE(gd, denom);
        TSTORE(go, dst);
    }
}

} // namespace supernpu::tile_isa
#endif
