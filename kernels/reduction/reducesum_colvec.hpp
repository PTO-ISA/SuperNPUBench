#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ==============================================
// ==============================================
template<typename tileSrc, typename tileSum>
void __vec__ reducesum_col_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileSum::TileDType __in__ old_sum
)
{
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);


//    typename tileSum::DType upd_sum = old_sum_ptr[i];
    typename tileSum::DType upd_sum = 0;

    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j+=8){
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;
        size_t src_idx_4 =  i * tileSrc::ColStride + (j + 4) * tileSrc::RowStride;
        size_t src_idx_5 =  i * tileSrc::ColStride + (j + 5) * tileSrc::RowStride;
        size_t src_idx_6 =  i * tileSrc::ColStride + (j + 6) * tileSrc::RowStride;
        size_t src_idx_7 =  i * tileSrc::ColStride + (j + 7) * tileSrc::RowStride;
        typename tileSum::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileSum::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename tileSum::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];
        typename tileSum::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];
        typename tileSum::DType sum_0123 = sum_01 + sum_23;
        typename tileSum::DType sum_4567 = sum_45 + sum_67;
        typename tileSum::DType sum_tmp = sum_0123 + sum_4567;
        upd_sum = upd_sum + sum_tmp;
    }

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_sum = upd_sum + src_ptr[src_idx];
    }
*/
    new_sum_ptr[i] = upd_sum + old_sum_ptr[i];
}




template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducesum_colsum_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,
    dtype *out_ptr
)
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; //



    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; //
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //
    using tile_shapeSum_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; //
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);
//    gm_shapeOut ZeroGm(inzero_ptr);
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeSum_row SumTile_row;
    tile_shapeSum_row oldSumTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeSum>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);
    itOut gOIter(out_ptr);

//    dtype zero = 0;

    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
        auto gO = gOIter(0, j);
        TEXPANDSCALAR(oldSumTile, 0);//初始化为0
//        TLOAD(oldSumTile, gZero);//初始化为0
        //初始化old_sum的tile
        //need
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            reducesum_col_kernel<tile_shapeData, tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
            oldSumTile = SumTile;
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_col, gI);
            reducesum_col_kernel<tile_shapeData_col,tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile_col.data(), oldSumTile.data());
            oldSumTile = SumTile;
        }
        TSTORE(gO, SumTile);
    }
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldSumTile_row, 0);//初始化为0
//        TLOAD(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            reducesum_col_kernel<tile_shapeData_row,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_row.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reducesum_col_kernel<tile_shapeData_cor,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_cor.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        TSTORE(gO, SumTile_row);
    }
}

#endif
