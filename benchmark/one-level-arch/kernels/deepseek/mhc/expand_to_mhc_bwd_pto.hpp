// =============================================================================
// expand_to_mhc_bwd_pto.hpp — MHC 扩展反向：沿 mhc 维求和归约（tile 版）
// =============================================================================
//
// 【功能】
//   前向 expand（沿 mhc 维复制）的反向：x_grad[n, j] = Σ_m o_grad[n, m, j]，沿 mhc 维归约。
//
// 【源端】TileKernels/tile_kernels/mhc/expand_kernel.py (expand_to_mhc_bwd_tl)
//
// 【迁移映射】
//   Σ_m o_grad[*, m, *]（沿 mhc 归约）→ TCOLSUM（按行求和→每列一个和）
//   bf16→fp32 累加           → TCVT 先升精度再 TCOLSUM（避免低精度累加误差）
//   fp32→bf16                → TCVT
//   T.copy                  → TLOAD / TSTORE
//
// 【约束】
//   - global_iterator 仅 2 参 → 3D(N,MHC,H) 展平为 2D(N*MHC, H)
//   - MhcMult 须 16 的倍数（bf16 tile_in 行方向 32B 对齐）；TileH 须 64 的倍数（1×TileH≥128B）
//
// 【算法步骤（每 token n, h-block th）】
//   1) TLOAD 取 (MhcMult×TileH) bf16 tile（token n 的所有 mhc 行）
//   2) TCVT 升到 fp32
//   3) TCOLSUM 沿行(mhc)归约 → (1×TileH) 每 j 上 Σ_m
//   4) TCVT 回 bf16；TSTORE 写 x_grad
//   注：与 fwd 正反对称——fwd 是 TLOAD+循环 TSTORE（广播写），bwd 是 TCVT+TCOLSUM（归约读）。
// =============================================================================
#ifndef SUPERNPU_EXPAND_TO_MHC_BWD_PTO_HPP
#define SUPERNPU_EXPAND_TO_MHC_BWD_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int Hidden, int MhcMult, int TileH = 64>
void expand_to_mhc_bwd(__bf16 *o_grad, __bf16 *x_grad) {
    static_assert(NumTokens > 0 && Hidden > 0 && MhcMult > 0, "dim must be positive");
    static_assert(MhcMult * 2 % 32 == 0, "MhcMult(bf16) must be multiple of 16");
    static_assert(TileH * 2 % 32 == 0, "TileH(bf16) must be 32B aligned");
    using namespace pto;
    using gm_in  = global_tensor<__bf16, RowMajor<NumTokens * MhcMult, Hidden>>; // 3D 展平 2D
    using gm_out = global_tensor<__bf16, RowMajor<NumTokens, Hidden>>;
    using tile_in  = Tile<Location::Vec, __bf16, MhcMult, TileH, BLayout::RowMajor>;
    using tile_f   = Tile<Location::Vec, float,   MhcMult, TileH, BLayout::RowMajor>;
    using tile_sum = Tile<Location::Vec, float, 1, TileH, BLayout::RowMajor, 1, TileH>;
    using tile_out = Tile<Location::Vec, __bf16, 1, TileH, BLayout::RowMajor, 1, TileH>;
    using it_in  = global_iterator<gm_in,  tile_in>;
    using it_out = global_iterator<gm_out, tile_out>;
    it_in in_iter(o_grad); it_out out_iter(x_grad);
    constexpr int kTH = Hidden / TileH;
    for (int n = 0; n < NumTokens; ++n) {
        for (int th = 0; th < kTH; ++th) {
            auto gin = in_iter(n, th); auto gout = out_iter(n, th);
            tile_in src; tile_f sf; tile_sum sumf; tile_out oq;
            TLOAD(src, gin);                            // 1) token n 的所有 mhc 行
            TCVT(sf, src);                              // 2) bf16→fp32
            TCOLSUM(sumf, sf);                         // 3) 沿 mhc(行) 归约 → 每 j 上 Σ_m
            TCVT(oq, sumf);                            // 4) fp32→bf16
            TSTORE(gout, oq);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
