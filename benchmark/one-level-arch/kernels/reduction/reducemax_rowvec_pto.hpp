#ifndef REDUCEMAXROWVEC_KERNEL_HPP
#define REDUCEMAXROWVEC_KERNEL_HPP

#pragma once
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reducemax_row_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    const int Mb = gIM / tM;
    const int Nb = gIN / tN;
    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, 1>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>;
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeMax_col = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, rmd_M, 1>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shapeTmp_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeTmp_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;
    using tile_shapeTmp_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeData_row dataTile_row;
    tile_shapeData_col dataTile_col;
    tile_shapeData_cor dataTile_cor;
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;
    tile_shapeMax_col MaxTile_col;
    tile_shapeMax_col oldMaxTile_col;
    tile_shapeTmp tmpTile;
    tile_shapeTmp_row tmpTile_row;
    tile_shapeTmp_col tmpTile_col;
    tile_shapeTmp_cor tmpTile_cor;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldMaxTile, static_cast<dtype>(0));

        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i);
            TLOAD(dataTile, gI);
            TROWMAX(MaxTile, dataTile);
            TMAX(oldMaxTile, oldMaxTile, MaxTile);
        }
        if constexpr (rmd_N > 0) {
            auto gI = gIIter(j, Nb);
            TLOAD(dataTile_row, gI);
            TROWMAX(MaxTile, dataTile_row);
            TMAX(oldMaxTile, oldMaxTile, MaxTile);
        }
        TSTORE(gO, oldMaxTile);
    }
    if constexpr (rmd_M > 0) {
        auto gO = gOIter(Mb, 0);
        TEXPANDSCALAR(oldMaxTile_col, static_cast<dtype>(0));

        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);
            TLOAD(dataTile_col, gI);
            TROWMAX(MaxTile_col, dataTile_col);
            TMAX(oldMaxTile_col, oldMaxTile_col, MaxTile_col);
        }
        if constexpr (rmd_N > 0) {
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            TROWMAX(MaxTile_col, dataTile_cor);
            TMAX(oldMaxTile_col, oldMaxTile_col, MaxTile_col);
        }
        TSTORE(gO, oldMaxTile_col);
    }
}

#endif
