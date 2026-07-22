// =============================================================================
// fused_weight_pto.hpp — Engram 权重融合 bf16×bf16→fp32（PTO 一层 tile 版）
// =============================================================================
//
// 【功能】
//   逐元素计算 weight_hidden × weight_embed，结果升到 fp32：
//     weight_fused[h, j] = (float)weight_hidden[h, j] * (float)weight_embed[h, j]
//   形状 (HcMult, Hidden)。用于 engram 门控的权重预处理。
//
// 【源端】TileKernels/tile_kernels/engram/engram_fused_weight_kernel.py
//   源端 32 threads × vec_size=8 的 SIMT warp 并行，bf16 直接乘落 fp32。
//
// 【迁移映射】
//   bf16→fp32 升精度        → TCVT(dst_fp32, src_bf16)
//   逐元素乘 a*b            → TMUL(dst, a, b)
//   T.copy(GM↔fragment)    → TLOAD / TSTORE
//   SIMT warp 并行           → tile 级并行（一次处理整 tile，非 SIMT）
//
// 【约束】
//   - TileW 须为 16 的倍数（bf16：1×TileW 列宽 32B 对齐）；且 1×TileW≥128B（TLOAD 最小）→ TileW≥64。
//   - Hidden 须被 TileW 整除。
//
// 【算法步骤（每 (h, tile)）】
//   1) TLOAD 取 weight_hidden 与 weight_embed 的 (1×TileW) bf16 tile
//   2) TCVT 各自升到 fp32
//   3) TMUL 逐元素乘出 fp32
//   4) TSTORE 写回 fp32 结果
// =============================================================================
#ifndef SUPERNPU_FUSED_WEIGHT_PTO_HPP
#define SUPERNPU_FUSED_WEIGHT_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int HcMult, int Hidden, int TileW = 64>
void fused_weight(__bf16 *weight_hidden, __bf16 *weight_embed, float *weight_fused) {
    static_assert(HcMult > 0 && Hidden > 0, "dim must be positive");
    static_assert(TileW * 2 % 32 == 0, "TileW(bf16) must be 32B aligned");
    constexpr int kTiles = Hidden / TileW;                // hidden 方向 tile 数
    using namespace pto;
    using gm_bf = global_tensor<__bf16, RowMajor<HcMult, Hidden>>;
    using gm_f  = global_tensor<float,   RowMajor<HcMult, Hidden>>;
    using tile_bf = Tile<Location::Vec, __bf16, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_bf = global_iterator<gm_bf, tile_bf>;
    using it_f  = global_iterator<gm_f,  tile_f>;
    it_bf h_iter(weight_hidden);                          // weight_hidden 迭代器
    it_bf e_iter(weight_embed);                           // weight_embed 迭代器
    it_f  o_iter(weight_fused);                           // 输出迭代器
    for (int h = 0; h < HcMult; ++h) {                    // hc_mult 维标量循环
        for (int t = 0; t < kTiles; ++t) {
            auto gh = h_iter(h, t); auto ge = e_iter(h, t); auto go = o_iter(h, t); // 左值绑定
            tile_bf a, b; tile_f af, bf, of;
            TLOAD(a, gh);                                 // 1a) weight_hidden bf16 tile
            TLOAD(b, ge);                                 // 1b) weight_embed  bf16 tile
            TCVT(af, a);                                  // 2a) bf16→fp32
            TCVT(bf, b);                                  // 2b) bf16→fp32
            TMUL(of, af, bf);                            // 3) 逐元素 fp32 乘
            TSTORE(go, of);                              // 4) 写回 fp32
        }
    }
}

} // namespace supernpu::tile_isa
#endif
