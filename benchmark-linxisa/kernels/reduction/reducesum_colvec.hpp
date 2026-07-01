#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
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
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeSum_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>;

    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeTmp_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeTmp_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeTmp_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;
    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeSum_row SumTile_row;
    tile_shapeSum_row oldSumTile_row;

    tile_shapeTmp tmpTile;
    tile_shapeTmp_col tmpTile_col;
    tile_shapeTmp_row tmpTile_row;
    tile_shapeTmp_cor tmpTile_cor;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    for (int j = 0; j < Nb; ++j) {
        auto gO = gOIter(0, j);
        TEXPANDS(oldSumTile, static_cast<dtype>(0.0f));

        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            TCOLSUM(SumTile, dataTile, tmpTile, /*isBinary=*/true);
            TADD(oldSumTile, oldSumTile, SumTile);
        }
        if constexpr (rmd_M > 0) {
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_col, gI);
            TCOLSUM(SumTile, dataTile_col, tmpTile_col, /*isBinary=*/true);
            TADD(oldSumTile, oldSumTile, SumTile);
        }
        TSTORE(gO, oldSumTile);
    }
    if constexpr (rmd_N > 0) {
        auto gO = gOIter(0, Nb);
        TEXPANDS(oldSumTile_row, static_cast<dtype>(0));

        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            TCOLSUM(SumTile_row, dataTile_row, tmpTile_row, /*isBinary=*/true);
            TADD(oldSumTile_row, oldSumTile_row, SumTile_row);
        }
        if constexpr (rmd_M > 0) {
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            TCOLSUM(SumTile_row, dataTile_cor, tmpTile_cor, /*isBinary=*/true);
            TADD(oldSumTile_row, oldSumTile_row, SumTile_row);
        }
        TSTORE(gO, oldSumTile_row);
    }
}

#endif
