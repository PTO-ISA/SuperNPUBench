// =============================================================================
// inplace_unique_group_indices_pto.hpp — 组索引原地去重（tile 版，pairwise 模拟）
// =============================================================================
//
// 【功能】
//   同 token 内重复组索引除首次外置 -1。
//
// 【源端】TileKernels/tile_kernels/moe/inplace_unique_group_indices_kernel.py
//   源端用 2×uint64 位图 + 串行扫描；TROWMIN/cumsum 工具链未提供，跨元素前缀难并行。
//
// 【迁移映射】
//   位图前缀难并行 → 改 O(NumTopk^2) pairwise（跨 token tile 并行）：
//     每 (a<b) 对: TEXTRACT 列 a/b(TCN×8 valid1) → TCMP(==) → TSEL(col_b=-1 命中) → TINSERT 回写
//   -1 处理幂等：-1 vs 真值不匹配，-1 vs -1 标记 -1 无害（见下注释）
//
// 【约束】NumTopk/TileN 须 8 的倍数（int32 32B 对齐）；支持任意 num_groups（无 128 上限）。
//
// 【算法步骤（每 tile tn）】
//   TLOAD idx; for b in 1..N-1: for a in 0..b-1:
//     TEXTRACT(col_a, idx, 0, a); TEXTRACT(col_b, idx, 0, b)
//     TCMP(eq, col_a, col_b); TSEL(col_b, eq, -1); TINSERT(idx, col_b, 0, b)
//   TSTORE idx
// =============================================================================
#ifndef SUPERNPU_INPLACE_UNIQUE_GROUP_INDICES_PTO_HPP
#define SUPERNPU_INPLACE_UNIQUE_GROUP_INDICES_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int NumTopk, int TileN = 8>
void inplace_unique_group_indices(std::int32_t *group_indices) {
    static_assert(NumTokens > 0 && NumTopk > 0, "dim must be positive");
    static_assert(NumTopk % 8 == 0 && TileN % 8 == 0, "NumTopk/TileN must be multiple of 8");
    constexpr int kTN = NumTokens / TileN;
    using namespace pto;
    using gm = global_tensor<std::int32_t, RowMajor<NumTokens, NumTopk>>;
    using tile_full = Tile<Location::Vec, std::int32_t, TileN, NumTopk, BLayout::RowMajor>;
    using tile_col = Tile<Location::Vec, std::int32_t, TileN, 8, BLayout::RowMajor, TileN, 1>; // 单列(每 token 一个)
    using it = global_iterator<gm, tile_full>;
    it idx_iter(group_indices);
    for (int tn = 0; tn < kTN; ++tn) {
        auto gidx = idx_iter(tn, 0);
        tile_full idx; TLOAD(idx, gidx);
        tile_col neg; TEXPANDS(neg, -1);
        for (int b = 1; b < NumTopk; ++b) {              // 逐列 b
            for (int a = 0; a < b; ++a) {                // 与每个更早列 a 比较
                tile_col ca, cb, eq;
                TEXTRACT(ca, idx, 0, a);                 // 提取列 a
                TEXTRACT(cb, idx, 0, b);                 // 提取列 b
                TCMP(eq, ca, cb);                         // 列 a==列 b?（EQ）
                TSEL(cb, eq, neg);                       // 命中处置 -1，否则保持
                TINSERT(idx, cb, 0, b);                  // 回写列 b
            }
        }
        auto gout = idx_iter(tn, 0);
        TSTORE(gout, idx);
    }
}

} // namespace supernpu::tile_isa
#endif
