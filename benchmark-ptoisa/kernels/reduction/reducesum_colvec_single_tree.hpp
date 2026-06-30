#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducesum_colsum_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    const int Mb = gIM / tM;
    const int Nb = gIN / tN;
    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, Mb, tN, BLayout::RowMajor>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeSum SumTile;
    tile_shapeTmpSum tmpSumTile;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);

    for (size_t i = 0; i < static_cast<size_t>(Mb); ++i) {
        auto gI = gIIter(i, 0);
        TLOAD(dataTile, gI);
        tile_shapeSum partialSum;
        TCOLSUM(partialSum, dataTile);

        using SingleRow = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
        SingleRow rowView;
        TMOV(rowView, partialSum);
        if (i == 0) {
            TMOV(tmpSumTile, partialSum);
        } else {
            TADD(tmpSumTile, tmpSumTile, partialSum);
        }
    }
    if constexpr (rmd_M > 0) {
        auto gI = gIIter(Mb, 0);
        TLOAD(dataTile_col, gI);
        tile_shapeSum partialSum;
        TCOLSUM(partialSum, dataTile_col);
        TADD(tmpSumTile, tmpSumTile, partialSum);
    }

    TCOLSUM(SumTile, tmpSumTile);
    TSTORE(gO, SumTile);
}

#endif
