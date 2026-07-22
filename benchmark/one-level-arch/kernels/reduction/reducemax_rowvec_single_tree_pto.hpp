#ifndef REDUCEMAXTROWMAX_KERNEL_HPP
#define REDUCEMAXTROWMAX_KERNEL_HPP

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
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>;
    using tile_shapeTmpMax = Tile<Location::Vec, dtype, tM, 64, BLayout::ColMajor, tM, Nb * 4>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeMax MaxTile;
    tile_shapeTmpMax oldtmpMaxTile;
    tile_shapeTmpMax tmpMaxTile;
    tile_shapeTmp tmpTile;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpMaxTile, static_cast<dtype>(0));

    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);
        TLOAD(dataTile, gI);
        tile_shapeMax partialMax;
        TROWMAX(partialMax, dataTile);
        TMAX(oldtmpMaxTile, oldtmpMaxTile, partialMax);
    }
    TCVT(tmpMaxTile, oldtmpMaxTile);

    TROWMAX(MaxTile, tmpMaxTile);
    TSTORE(gO, MaxTile);
}

#endif
