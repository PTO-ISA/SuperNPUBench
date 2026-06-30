#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, int gIM, int gIN, int tM, int tN, int tM_VLD>
void reducesum_colsum_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM / 8, gIN * 8>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM / 8, tN * 8, BLayout::RowMajor, tM_VLD / 8, tN * 8>;
    using tile_shapeTmp = Tile<Location::Vec, dtype, 1, tN * 8, BLayout::RowMajor>;
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN * 8, BLayout::RowMajor, 1, tN>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeTmp TmpTile;
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);
    TEXPANDS(oldSumTile, static_cast<dtype>(0));
    auto gI = gIIter(0, 0);
    TLOAD(dataTile, gI);
    TCOLSUM(TmpTile, dataTile);
    TCOLSUM(SumTile, TmpTile);
    TADD(oldSumTile, oldSumTile, SumTile);
    TSTORE(gO, SumTile);
}

#endif
