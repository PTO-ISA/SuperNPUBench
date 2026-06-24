#ifndef CUMSUMCOL_KERNEL_HPP
#define CUMSUMCOL_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ==============================================
// ==============================================



template<typename tileData, typename tileSum>
void __vec__ cumsum_col_kernel(
    typename tileSum::TileDType __out__ new_sum,
    typename tileData::TileDType __out__ out,
    const typename tileData::TileDType __in__ src,
    const typename tileSum::TileDType __in__ old_sum
)
{
    size_t i = blkv_get_index_x();
    size_t sum_idx = i * tileSum::RowStride;

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileData::DType *out_ptr = blkv_get_tile_ptr(out);
    __vbuf__ typename tileData::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);


    typename tileSum::DType upd_sum = old_sum_ptr[i];
//    printf("upd_sum = %d",upd_sum);
//    typename tileData::DType upd_out = old_sum_ptr[i];
/*
    for(size_t j=0;j<tileSrc::ValidRow;j+=4){
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;
        typename tileSum::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileSum::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename tileSum::DType sum_0123 = sum_01 + sum_23;
        upd_sum = upd_sum + sum_0123;
    }
*/
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileData::ValidRow;j++){
        size_t idx =  i * tileData::ColStride + j * tileData::RowStride;
        typename tileData::DType sum_out = upd_sum + src_ptr[idx];
        upd_sum = sum_out;
        out_ptr[idx] = static_cast<typename tileData::DType>(sum_out);
    }
    new_sum_ptr[i] = upd_sum;
}




template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void cumsum_col_rand(
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

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //将gm中的Tensor先声明为一维数据
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, gIN>>;
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
    tile_shapeData OutTile;
    tile_shapeData_col dataTile_col;
    tile_shapeData_col OutTile_col;
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_row OutTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeData_cor OutTile_cor;

    tile_shapeSum_row SumTile_row;
    tile_shapeSum_row oldSumTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
//    using itZero = global_iterator<gm_shapeOut, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
//    using itSum = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

//    dtype zero = 0;

    for (int j = 0; j < Nb; ++j) {
        TEXPANDSCALAR(oldSumTile, 0);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            auto gO = gOIter(i, j);
            TLOAD(dataTile, gI);
//            printf("in0 : %d, %d\n",in_ptr[i*tM], i*tM);
            cumsum_col_kernel<tile_shapeData, tile_shapeSum><<<tile_shapeData::ValidCol, 1, 1>>>(SumTile.data(), OutTile.data(), dataTile.data(), oldSumTile.data());
            oldSumTile = SumTile;
            TSTORE(gO, OutTile);
//            printf("out0 : %d,%d\n", out_ptr[i*tM],i*tM);
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, j);
            auto gO = gOIter(Mb, j);
            TLOAD(dataTile_col, gI);
            cumsum_col_kernel<tile_shapeData_col,tile_shapeSum><<<tile_shapeData_col::ValidCol, 1, 1>>>(SumTile.data(), OutTile_col.data(), dataTile_col.data(), oldSumTile.data());
            oldSumTile = SumTile;
            TSTORE(gO, OutTile_col);
        }
//        TSTORE(gO, SumTile);
    }
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);
//        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldSumTile_row, 0);//初始化为0
//        TLOAD(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            auto gO = gOIter(i, Nb);
            TLOAD(dataTile_row, gI);
            cumsum_col_kernel<tile_shapeData_row,tile_shapeSum_row><<<tile_shapeData_row::ValidCol, 1, 1>>>(SumTile_row.data(), OutTile_row.data(), dataTile_row.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
            TSTORE(gO, OutTile_row);
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, Nb);
            auto gO = gOIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            cumsum_col_kernel<tile_shapeData_cor,tile_shapeSum_row><<<tile_shapeData_cor::ValidCol, 1, 1>>>(SumTile_row.data(), OutTile_cor.data(), dataTile_cor.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
            TSTORE(gO, OutTile_cor);
        }
    }
/*
    for(int i = 0; i < gIN; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
}



#endif