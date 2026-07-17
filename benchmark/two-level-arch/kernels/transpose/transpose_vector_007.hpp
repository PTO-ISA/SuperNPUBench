#include <common/pto_tileop.hpp>
#include "gemm.hpp"

using namespace pto;

//AI/IA = A, placeholder
//like Ttrans
template<typename dtype, typename tileInData, typename tileOutData>
void __vec__ transpose_007_impl(
    typename tileOutData::TileDType __out__ out,
    const typename tileInData::TileDType __in__ in
)
{
    size_t i = blkv_get_index_x(); // 4096
    size_t j = blkv_get_index_y(); // 3

    size_t idx_src = i * tileOutData::ValidRow + j;
    size_t idx_dst = i * tileOutData::ColStride + j * tileOutData::RowStride;
    blkv_get_tile_ptr(out)[idx_dst] = blkv_get_tile_ptr(in)[idx_src];
}


// output = input * A^T + bias ;;; input = [batch_size, in_features] , A = [out_features, in_features]
// A should be row majo

// y = x1^T.A.x2 + b, where x1=1xd1, x2=1xd2, A=doxd1xd2
// in1[DimIn1, 1] in2 [DimIn2, 1] bias[DimOut, 1] weight [DimOut, DimIn1, DimIn2]
template<typename dtype>
void transpose_007(
        dtype *out_ptr,
        dtype *in_ptr
)
{
    const int Mb = 4096 / 4096;

    using gm_shapeIn = global_tensor<dtype, RowMajor<1, 4096*3>>;     //将gm中的Tensor先声明为一维数据
    using gm_shapeOut = global_tensor<dtype, RowMajor<3, 4096>>;
    using tile_shapeInData = Tile<Location::Vec, dtype, 1, 4096*4, BLayout::RowMajor, 1, 4096*3>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeOutData = Tile<Location::Vec, dtype, 4, 4096, BLayout::RowMajor, 3, 4096>; // todo 尾块怎么处理？是否要作为参数写在这

    using itIn = global_iterator<gm_shapeIn, tile_shapeInData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeOutData>;

    tile_shapeInData InDataTile;
    tile_shapeOutData OutDataTile;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);
        TLOAD(InDataTile, gI);
        transpose_007_impl<dtype, tile_shapeInData, tile_shapeOutData><<<tile_shapeOutData::ValidCol, tile_shapeOutData::ValidRow, 1>>>(OutDataTile.data(), InDataTile.data());
        TSTORE(gO, OutDataTile);
    }

}

