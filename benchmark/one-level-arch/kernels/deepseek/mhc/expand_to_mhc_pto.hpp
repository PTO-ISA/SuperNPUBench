// =============================================================================
// expand_to_mhc_pto.hpp — MHC token 扩展（前向）：(N,H)→(N,MHC,H)（tile 版）
// =============================================================================
//
// 【功能】
//   把每个 token 的 hidden 向量沿新增 mhc 维复制 MhcMult 份：
//     o[i, m, j] = x[i, j]
//
// 【源端】TileKernels/tile_kernels/mhc/expand_kernel.py (expand_to_mhc_fwd_tl)
//
// 【迁移映射】
//   "广播写"（同一 tile 写到多个位置）→ 输入 tile 循环外只 TLOAD 一次，
//     m 维标量循环内重复 TSTORE 到 m 个输出位置
//   T.copy                → TLOAD / TSTORE
//
// 【约束】
//   - global_iterator 仅 2 参 → 输出 3D(N,MHC,H) 展平为 2D(N*MHC, H)
//   - 逐 token 处理（1×TileH tile）：TileH 须 64 的倍数（bf16 1×TileH≥128B 最小限制）
//
// 【算法步骤（每 token n, h-block th）】
//   1) TLOAD 取输入 (1×TileH) tile（循环外只取一次）
//   2) for m in 0..MhcMult-1: TSTORE 同一 tile 到输出 (n*MHC+m, th) 位置
// =============================================================================
#ifndef SUPERNPU_EXPAND_TO_MHC_PTO_HPP
#define SUPERNPU_EXPAND_TO_MHC_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int Hidden, int MhcMult, int TileH = 64>
void expand_to_mhc_fwd(__bf16 *x, __bf16 *o) {
    static_assert(NumTokens > 0 && Hidden > 0 && MhcMult > 0, "dim must be positive");
    static_assert(TileH * 2 % 32 == 0 && TileH >= 64, "TileH(bf16) must be >=64 and 32B aligned");
    constexpr int kTH = Hidden / TileH;
    using namespace pto;
    using gm_in  = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using gm_out = global_tensor<__bf16, RowMajor<NumTokens * MhcMult, Hidden>>; // 3D 展平 2D
    using tile_d = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor>;
    using it_in  = global_iterator<gm_in,  tile_d>;
    using it_out = global_iterator<gm_out, tile_d>;
    it_in in_iter(x); it_out out_iter(o);
    for (int n = 0; n < NumTokens; ++n) {                // 逐 token
        for (int th = 0; th < kTH; ++th) {
            auto gin = in_iter(n, th);
            tile_d src; TLOAD(src, gin);                // 1) 输入 tile 只 load 一次
            for (int m = 0; m < MhcMult; ++m) {          // 2) 沿 mhc 维复制
                auto gout = out_iter(n * MhcMult + m, th);
                TSTORE(gout, src);                       //    写到 m 个输出位置
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
