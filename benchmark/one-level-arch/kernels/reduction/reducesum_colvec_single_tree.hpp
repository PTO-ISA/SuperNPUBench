#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// ============================================================================
// single_tree 列归约 (column reduction): gIM×gIN → 1×gIN
//
// 策略 (对应 reference 的 reducesum_colvec_single_tree):
//   Phase 1: 对 Mb 个 tile 各做 TCOLSUM(tM×tN → 1×tN),
//            用 TINSERT 将每个 partial sum 存入 tmpSumTile 的第 i 行。
//   Phase 2: 对 tmpSumTile (Mb×tN) 做一次 TCOLSUM, 完成 Mb→1 最终归约。
//
// 与 basic 版本的区别:
//   basic: Mb 次 TCOLSUM + Mb 次顺序 TADD (依赖链)
//   single_tree: Mb 次 TCOLSUM + TINSERT (独立) + 1 次 TCOLSUM (树归约)
// ============================================================================

template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducesum_colsum_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    constexpr int Mb = gIM / tM;
    constexpr int Nb = gIN / tN;
    constexpr int rmd_M = gIM % tM;
    constexpr int rmd_N = gIN % tN;
    constexpr int totalPartials = Mb + (rmd_M > 0 ? 1 : 0);

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    // tmpSumTile: 存储 totalPartials 个 partial sum, 每行一个
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, totalPartials, tN, BLayout::RowMajor>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeTmp_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeTmp_final = Tile<Location::Vec, dtype, totalPartials, tN, BLayout::RowMajor>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeSum SumTile;
    tile_shapeSum partialSum;
    tile_shapeTmpSum tmpSumTile;
    tile_shapeTmp tmpTile;
    tile_shapeTmp_col tmpTile_col;
    tile_shapeTmp_final tmpTile_final;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);

    // Phase 1: 逐 tile 归约, 用 TINSERT 存储 partial sum
    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(i, 0);
        TLOAD(dataTile, gI);
        TCOLSUM(partialSum, dataTile, tmpTile, /*isBinary=*/true);
        TINSERT(tmpSumTile, partialSum, static_cast<uint16_t>(i), 0);
    }
    if constexpr (rmd_M > 0) {
        auto gI = gIIter(Mb, 0);
        TLOAD(dataTile_col, gI);
        TCOLSUM(partialSum, dataTile_col, tmpTile_col, /*isBinary=*/true);
        TINSERT(tmpSumTile, partialSum, static_cast<uint16_t>(Mb), 0);
    }

    // Phase 2: 对所有 partial sum 做最终树归约
    TCOLSUM(SumTile, tmpSumTile, tmpTile_final, /*isBinary=*/true);
    TSTORE(gO, SumTile);
}

#endif
