#ifndef REDUCEMAXTROWMAX_KERNEL_HPP
#define REDUCEMAXTROWMAX_KERNEL_HPP

#pragma once
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// ============================================================================
// single_tree 行归约 (row max reduction): gIM×gIN → gIM×1
//
// 策略 (对应 reference 的 reducemax_rowvec_single_tree):
//   Phase 1: 对 Nb 个 tile 各做 TROWMAX(tM×tN → tM×1),
//            用 TINSERT 将每个 partial max 存入 tmpSumTile 的第 i 列。
//   Phase 2: 对 tmpSumTile (tM×Nb) 做一次 TROWMAX, 完成 Nb→1 最终归约。
//
// 与 basic 版本的区别:
//   basic: Nb 次 TROWMAX + Nb 次顺序 TMAX (依赖链)
//   single_tree: Nb 次 TROWMAX + TINSERT (独立) + 1 次 TROWMAX (树归约)
// ============================================================================

template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reducemax_row_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    constexpr int Mb = gIM / tM;
    constexpr int Nb = gIN / tN;
    constexpr int rmd_M = gIM % tM;
    constexpr int rmd_N = gIN % tN;

    // partial max: tM×1 RowMajor (TROWMAX dst 约束: Cols==1)
    // tmpSumTile physical Cols 须满足: 32 字节对齐 + 2 的次幂
    constexpr int minAlignCols = (32 + sizeof(dtype) - 1) / sizeof(dtype);
    constexpr int rawCols = (Nb > minAlignCols) ? Nb : minAlignCols;
    constexpr int tmpCols = []() {
        int r = 1;
        while (r < rawCols) r <<= 1;
        return r;
    }();

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, 1>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor>;
    using tile_shapeTmpMax = Tile<Location::Vec, dtype, tM, tmpCols, BLayout::RowMajor, tM, Nb>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeTmp_final = Tile<Location::Vec, dtype, tM, tmpCols, BLayout::RowMajor>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeMax MaxTile;
    tile_shapeMax partialMax;
    tile_shapeTmpMax tmpMaxTile;
    tile_shapeTmp tmpTile;
    tile_shapeTmp_final tmpTile_final;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);

    // Phase 1: 逐 tile 行归约, 用 TINSERT 存储 partial max
    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);
        TLOAD(dataTile, gI);
        TROWMAX(partialMax, dataTile);
        TINSERT(tmpMaxTile, partialMax, 0, static_cast<uint16_t>(i));
    }

    // Phase 2: 对所有 partial max 做最终树归约
    TROWMAX(MaxTile, tmpMaxTile);
    TSTORE(gO, MaxTile);
}

#endif
