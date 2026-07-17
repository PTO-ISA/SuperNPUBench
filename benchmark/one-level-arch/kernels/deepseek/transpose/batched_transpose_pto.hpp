// =============================================================================
// batched_transpose_pto.hpp — 批量 3D 转置 (B,M,N)→(B,N,M)（tile 版）
// =============================================================================
//
// 【功能】
//   batched_transpose(x): (B, M, N) → (B, N, M)；transpose(x) 2D 是 B=1 特例。
//
// 【源端】TileKernels/tile_kernels/transpose/batched_transpose_kernel.py
//   源端用寄存器 4×4 转置 + swizzle shared memory 消 bank conflict（GPU SIMT 优化）。
//
// 【迁移映射】
//   手写转置+swizzle → 硬件原生 TTRANS(dst, src) 单指令（无需 swizzle）
//   T.copy → TLOAD / TSTORE
//   global_iterator 仅 2 参 → 3D(B,M,N) 展平为 2D(B*M, N)，输出 2D(B*N, M)
//
// 【约束】TileRows/Cols 使 dtype 32B 对齐。2D 已由 kernels/transpose/transpose_pto.hpp 实现。
//
// 【算法步骤（每 batch b, 行tile rt, 列tile ct）】
//   TLOAD src(2D b*Rows/TileRows+rt, ct)→TTRANS(dst, src)→TSTORE(2D b*Cols/TileCols+ct, rt)
// =============================================================================
#ifndef SUPERNPU_BATCHED_TRANSPOSE_PTO_HPP
#define SUPERNPU_BATCHED_TRANSPOSE_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

#ifndef SUPERNPU_PTO_TTRANS_NEEDS_TMP
#define SUPERNPU_PTO_TTRANS_NEEDS_TMP 0
#endif

namespace supernpu::tile_isa {

template <typename DType, int Batch, int Rows, int Cols,
          int TileRows = 16, int TileCols = 16>
void batched_transpose(DType *input, DType *output) {
    static_assert(Batch > 0 && Rows > 0 && Cols > 0, "dim must be positive");
    static_assert(TileRows * static_cast<int>(sizeof(DType)) % 32 == 0, "TileRows 须 32B 对齐");
    static_assert(TileCols * static_cast<int>(sizeof(DType)) % 32 == 0, "TileCols 须 32B 对齐");
    constexpr int kRowTiles = Rows / TileRows, kColTiles = Cols / TileCols;
    using namespace pto;
    // global_iterator 仅 2 参，3D(B,M,N) 展平为 2D(B*M, N)
    using gm_in  = global_tensor<DType, RowMajor<Batch * Rows, Cols>>;
    using gm_out = global_tensor<DType, RowMajor<Batch * Cols, Rows>>;
    using tile_s = Tile<Location::Vec, DType, TileRows, TileCols, BLayout::RowMajor>;
    using tile_d = Tile<Location::Vec, DType, TileCols, TileRows, BLayout::RowMajor>; // 转置后形状
    using it_in  = global_iterator<gm_in,  tile_s>;
    using it_out = global_iterator<gm_out, tile_d>;
    it_in in_iter(input); it_out out_iter(output);
    for (int b = 0; b < Batch; ++b) {
        for (int rt = 0; rt < kRowTiles; ++rt) {
            for (int ct = 0; ct < kColTiles; ++ct) {
                auto gi = in_iter(b * kRowTiles + rt, ct); auto go = out_iter(b * kColTiles + ct, rt);
                tile_s src; tile_d dst;
                TLOAD(src, gi);                           // GM → tile
#if SUPERNPU_PTO_TTRANS_NEEDS_TMP
                tile_s tmp; TTRANS(dst, src, tmp);        // 旧 API 需 tmp
#else
                TTRANS(dst, src);                         // 硬件转置 TileRows×TileCols→TileCols×TileRows
#endif
                TSTORE(go, dst);                          // tile → GM（行列互换位置）
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
