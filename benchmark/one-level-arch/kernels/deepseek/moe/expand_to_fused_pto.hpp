// =============================================================================
// expand_to_fused_pto.hpp — MoE token 扩展到 fused expert 布局（tile 版）
// =============================================================================
//
// 【功能】
//   把 (num_tokens, hidden) 按 token_topk_to_pos 复制到 num_topk 个扩展位置；
//   pos_to_expert<0 的位置清零。本质"scatter 写"。
//
// 【源端】TileKernels/tile_kernels/moe/expand_to_fused_kernel.py
//
// 【迁移映射】
//   "scatter 写"（同一输入 tile 写到多个目标行）→ 输入 tile 循环外 TLOAD 一次，
//     k 维标量循环内 TSTORE 到 pos 偏移的 global_tensor 视图
//   无效位置清零 → TEXPANDS(0) + TSTORE
//   T.copy → TLOAD / TSTORE
//
// 【约束】TileW 使 dtype 32B 对齐且 1×TileW≥128B（TLOAD 最小）。
//
// 【算法步骤】
//   1) 无效位置(pos_to_expert<0) 清零: TEXPANDS(0)→TSTORE
//   2) 每 token 每 tile: TLOAD src（循环外一次）→ for k: TSTORE 到 (pos, t)
// =============================================================================
#ifndef SUPERNPU_EXPAND_TO_FUSED_PTO_HPP
#define SUPERNPU_EXPAND_TO_FUSED_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename DType, int NumTokens, int Hidden, int NumTopk,
          int NumExpanded, int TileW = 16>
void expand_to_fused(DType *x, std::int32_t *token_topk_to_pos,
                     std::int32_t *pos_to_expert, DType *expanded_x) {
    static_assert(NumTokens > 0 && Hidden > 0 && NumTopk > 0 && NumExpanded > 0, "dim must be positive");
    static_assert(TileW * static_cast<int>(sizeof(DType)) % 32 == 0, "TileW*dtype 须 32B 对齐");
    constexpr int kTiles = Hidden / TileW;
    using namespace pto;
    using gm_in  = global_tensor<DType, RowMajor<NumTokens, Hidden>>;
    using gm_out = global_tensor<DType, RowMajor<NumExpanded, Hidden>>;
    using tile_d = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using it_in  = global_iterator<gm_in,  tile_d>;
    using it_out = global_iterator<gm_out, tile_d>;
    it_in in_iter(x); it_out out_iter(expanded_x);

    // 1) 无效位置清零
    tile_d zero; TEXPANDS(zero, static_cast<DType>(0));
    for (int p = 0; p < NumExpanded; ++p) {
        if (pos_to_expert[p] < 0)
            for (int t = 0; t < kTiles; ++t) { auto g = out_iter(p, t); TSTORE(g, zero); }
    }
    // 2) 每 token 复制到 num_topk 个目标位置
    for (int n = 0; n < NumTokens; ++n) {
        for (int t = 0; t < kTiles; ++t) {
            auto gin = in_iter(n, t);
            tile_d src; TLOAD(src, gin);               // 输入 tile 只 load 一次
            for (int k = 0; k < NumTopk; ++k) {
                int pos = token_topk_to_pos[n * NumTopk + k];
                if (pos >= 0) { auto g = out_iter(pos, t); TSTORE(g, src); }
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
