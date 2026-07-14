#ifndef REDUCESUMTROWSUM_KERNEL_HPP
#define REDUCESUMTROWSUM_KERNEL_HPP

#pragma once
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// ============================================================================
// single_tree 行归约 (row reduction): gIM×gIN → gIM×1
//
// 策略 (对应 reference 的 reducesum_rowvec_single_tree):
//   Phase 1: 对 Nb 个 tile 各做 TROWSUM(tM×tN → tM×1),
//            用 TINSERT 将每个 partial sum 存入 tmpSumTile 的第 i 列。
//   Phase 2: 对 tmpSumTile (tM×Nb) 做一次 TROWSUM, 完成 Nb→1 最终归约。
//
// 与 basic 版本的区别:
//   basic: Nb 次 TROWSUM + Nb 次顺序 TADD (依赖链)
//   single_tree: Nb 次 TROWSUM + TINSERT (独立) + 1 次 TROWSUM (树归约)
//
// Tile shape 设计:
//   - partialSum: tM×1 RowMajor (TROWSUM dst 约束: Cols==1, RowMajor 或 ColMajor)
//   - tmpSumTile: tM×tmpCols RowMajor, valid tM×Nb
//     tmpCols = nextPow2(max(Nb, ceil(32/sizeof(dtype))))
//     物理Cols须满足 32 字节对齐且为 2 的次幂
//   - TINSERT 在 col=i 处写入 1 列, 落在 valid region [0, Nb) 内
//   - 最终 TROWSUM 对 valid Cols=Nb 做树归约
// ============================================================================

template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reducesum_trowsum_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    constexpr int Mb = gIM / tM;
    constexpr int Nb = gIN / tN;
    constexpr int rmd_M = gIM % tM;
    constexpr int rmd_N = gIN % tN;

    // partial sum physical Cols=1 (TROWSUM dst 约束允许 Cols==1)
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
    using tile_shapeSum = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor>;
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, tM, tmpCols, BLayout::RowMajor, tM, Nb>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeTmp_final = Tile<Location::Vec, dtype, tM, tmpCols, BLayout::RowMajor>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeSum SumTile;
    tile_shapeSum partialSum;
    tile_shapeTmpSum tmpSumTile;
    tile_shapeTmp tmpTile;
    tile_shapeTmp_final tmpTile_final;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);

    // Phase 1: 逐 tile 行归约, 用 TINSERT 存储 partial sum
    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);
        TLOAD(dataTile, gI);
        TROWSUM(partialSum, dataTile);
        TINSERT(tmpSumTile, partialSum, 0, static_cast<uint16_t>(i));
    }

    // Phase 2: 对所有 partial sum 做最终树归约
    TROWSUM(SumTile, tmpSumTile);
    TSTORE(gO, SumTile);
}

#endif
