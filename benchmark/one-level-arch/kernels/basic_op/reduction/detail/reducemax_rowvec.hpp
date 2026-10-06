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
    // PTO permits larger reduction sources, but the current timing model
    // deadlocks on this benchmark's 16 KiB TROWMAX source.  Decompose the
    // same ordered row reduction into smaller, independently legal tiles;
    // TMAX then combines the partial maxima without changing the result.
    constexpr int kReductionSourceBytes = 2048;
    constexpr int kReduceNCapacity =
        kReductionSourceBytes / (tM * static_cast<int>(sizeof(dtype)));
    constexpr int kReduceN = tN < kReduceNCapacity ? tN : kReduceNCapacity;
    static_assert(kReduceN > 0, "tM is too large for the selected reduction chunk");
    static_assert(tN % kReduceN == 0,
                  "logical tN must be divisible by the ISA reduction width");

    const int Mb = gIM / tM;
    const int Nb = gIN / kReduceN;
    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % kReduceN;

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, 1>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, kReduceN, BLayout::RowMajor>;
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, kReduceN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor, tM, 1>;
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, kReduceN, BLayout::RowMajor, rmd_M, kReduceN>;
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, kReduceN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeMax_col = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor, rmd_M, 1>;

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

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    #pragma clang loop unroll(full)
    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        // Seed from real data.  Zero is not max's identity for negative rows.
        auto gI0 = gIIter(j, 0);
        TLOAD(dataTile, gI0);
        TROWMAX(oldMaxTile, dataTile);

        #pragma clang loop unroll(full)
        for (int i = 1; i < Nb; ++i) {
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
        auto gI0 = gIIter(Mb, 0);
        TLOAD(dataTile_col, gI0);
        TROWMAX(oldMaxTile_col, dataTile_col);

        #pragma clang loop unroll(full)
        for (int i = 1; i < Nb; ++i) {
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
