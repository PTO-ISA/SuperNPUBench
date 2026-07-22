#include <common/pto_tileop.hpp>
#include "gemm.hpp"

using namespace pto;


template<typename dtype, typename tileData>
void __vec__ transpose_050_impl(
    typename tileData::TileDType __out__ out,
    const typename tileData::TileDType __in__ in
)
{
    size_t i = blkv_get_index_x(); // y
    size_t j = blkv_get_index_y(); // x 4096*3

    size_t row_out = j / 8;
    size_t col_out = j % 8;

    size_t idx_in = col_out * 64 + row_out;

    size_t idx_src = i * tileData::ColStride + idx_in * tileData::RowStride;
    size_t idx_dst = i * tileData::ColStride + j * tileData::RowStride;
    blkv_get_tile_ptr(out)[idx_dst] = blkv_get_tile_ptr(in)[idx_src];
}


template<typename dtype>
void transpose_050(
        dtype *out_ptr,
        dtype *in_ptr
)
{


    using gm_shapeIn = global_tensor<dtype, RowMajor<512, 64>>;     //将gm中的Tensor先声明为一维数据
    using gm_shapeOut = global_tensor<dtype, RowMajor<512, 64>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 512, 64, BLayout::RowMajor, 512, 64>; // todo 尾块怎么处理？是否要作为参数写在这

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;

    tile_shapeData InDataTile;
    tile_shapeData OutDataTile;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);


    auto gI = gIIter(0, 0);
    auto gO = gOIter(0, 0);
    TLOAD(InDataTile, gI);
    transpose_050_impl<dtype, tile_shapeData><<<tile_shapeData::ValidCol, tile_shapeData::ValidRow, 1>>>(OutDataTile.data(), InDataTile.data());
    TSTORE(gO, OutDataTile);


}


