// =============================================================================
// reduce_fused_pto.hpp — MoE 加权归约回 token 级（tile 版）
// =============================================================================
//
// 【功能】
//   out[token] = Σ_k weight[token,k] * x[pos[token,k]]    （加权求和，可选 ×sf、×x_sf）
//
// 【源端】TileKernels/tile_kernels/moe/reduce_fused_kernel.py
//
// 【迁移映射】
//   TFMA 工具链未提供 → TMUL(x*weight) + TADD(累加)（融合乘加拆两步）
//   acc 初值清零 → TEXPANDS(0)
//   bf16→fp32 累加 → TCVT 先升精度（避免低精度累加误差）
//   T.copy → TLOAD / TSTORE
//
// 【约束】TileW 须 8 的倍数（fp32 32B 对齐）；1×TileW≥128B（TLOAD 最小）。
//
// 【算法步骤（每 token n, h-tile t）】
//   acc=0; for k in 0..NumTopk-1:
//     TLOAD xq(pos)→TCVT xf→TMULS(wf, xf, weight[k])→TADD(acc, acc, wf)
//   TCVT(oq, acc)→TSTORE
// =============================================================================
#ifndef SUPERNPU_REDUCE_FUSED_PTO_HPP
#define SUPERNPU_REDUCE_FUSED_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename DTypeIn, typename DTypeOut, int NumTokens, int Hidden,
          int NumTopk, int NumExpanded, int TileW = 16>
void reduce_fused(DTypeIn *x, float *topk_weights,
                  std::int32_t *token_topk_to_pos, DTypeOut *out) {
    static_assert(NumTokens > 0 && Hidden > 0 && NumTopk > 0 && NumExpanded > 0, "dim must be positive");
    static_assert(TileW % 8 == 0, "TileW must be multiple of 8");
    constexpr int kTiles = Hidden / TileW;
    using namespace pto;
    using gm_in  = global_tensor<DTypeIn,  RowMajor<NumExpanded, Hidden>>;
    using gm_out = global_tensor<DTypeOut, RowMajor<NumTokens,  Hidden>>;
    using tile_in = Tile<Location::Vec, DTypeIn,  1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    1, TileW, BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, DTypeOut, 1, TileW, BLayout::RowMajor>;
    using it_in = global_iterator<gm_in,  tile_in>;
    using it_out = global_iterator<gm_out, tile_o>;
    it_in in_iter(x); it_out out_iter(out);
    for (int n = 0; n < NumTokens; ++n) {
        for (int t = 0; t < kTiles; ++t) {
            tile_f acc, xf, wf; tile_in xq; tile_o oq;
            TEXPANDS(acc, 0.0f);                          // acc 初值 0
            for (int k = 0; k < NumTopk; ++k) {
                int pos = token_topk_to_pos[n * NumTopk + k];
                if (pos >= 0) {
                    float w = topk_weights[n * NumTopk + k];
                    auto gx = in_iter(pos, t);
                    TLOAD(xq, gx);                       // 取 pos 行的 tile
                    TCVT(xf, xq);                         // → fp32
                    TMULS(wf, xf, w);                     // x * weight
                    TADD(acc, acc, wf);                   // 累加（替代 TFMA）
                }
            }
            TCVT(oq, acc);                                // → 输出 dtype
            auto gout = out_iter(n, t);
            TSTORE(gout, oq);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
