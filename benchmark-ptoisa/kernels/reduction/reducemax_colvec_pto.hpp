#ifndef REDUCEMAXCOLVEC_KERNEL_HPP
#define REDUCEMAXCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducemax_col_rand(
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
    using tile_shapeMax = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeMax_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;
    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeMax_row MaxTile_row;
    tile_shapeMax_row oldMaxTile_row;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    for (int j = 0; j < Nb; ++j) {
        auto gO = gOIter(0, j);
        TEXPANDS(oldMaxTile, static_cast<dtype>(0));

        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            TCOLMAX(MaxTile, dataTile);
            TMAX(oldMaxTile, oldMaxTile, MaxTile);
        }
        if constexpr (rmd_M > 0) {
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_col, gI);
            TCOLMAX(MaxTile, dataTile_col);
            TMAX(oldMaxTile, oldMaxTile, MaxTile);
        }
        TSTORE(gO, oldMaxTile);
    }
    if constexpr (rmd_N > 0) {
        auto gO = gOIter(0, Nb);
        TEXPANDS(oldMaxTile_row, static_cast<dtype>(0));

        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            TCOLMAX(MaxTile_row, dataTile_row);
            TMAX(oldMaxTile_row, oldMaxTile_row, MaxTile_row);
        }
        if constexpr (rmd_M > 0) {
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            TCOLMAX(MaxTile_row, dataTile_cor);
            TMAX(oldMaxTile_row, oldMaxTile_row, MaxTile_row);
        }
        TSTORE(gO, oldMaxTile_row);
    }
}

#endif
