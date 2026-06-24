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
template<typename tileSrc, typename tileMax>
void __vec__ reducemax_col_kernel(
    typename tileMax::TileDType __out__ new_max,
    const typename tileSrc::TileDType __in__ src,
    const typename tileMax::TileDType __in__ old_max
)
{
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);


    typename tileMax::DType upd_max = old_max_ptr[i];
/*
    for(size_t j=0;j<tileSrc::ValidRow;j+=4){
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;
        typename tileMax::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileMax::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename tileMax::DType sum_0123 = sum_01 + sum_23;
        upd_sum = upd_sum + sum_0123;
    }
*/
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
        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        typename tileMax::DType max_45 = blkv_max(src_ptr[src_idx_4], src_ptr[src_idx_5]);
        typename tileMax::DType max_67 = blkv_max(src_ptr[src_idx_6], src_ptr[src_idx_7]);
        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        typename tileMax::DType max_4567 = blkv_max(max_45, max_67);
        typename tileMax::DType max_tmp = blkv_max(max_0123, max_4567);
        upd_max = blkv_max(upd_max, max_tmp);
    }

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_max = blkv_max(upd_max, src_ptr[src_idx]);
//        upd_max = upd_sum + src_ptr[src_idx];
    }
*/
    new_max_ptr[i] = upd_max;
}



template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducemax_col_rand(
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
    using tile_shapeMax = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; //



    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; //
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //
    using tile_shapeMax_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; //
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);
//    gm_shapeOut ZeroGm(inzero_ptr);
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeMax_row MaxTile_row;
    tile_shapeMax_row oldMaxTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeMax>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);
    itOut gOIter(out_ptr);

//    dtype zero = 0;

//    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldMaxTile, 0);//初始化为0
//        TLOAD(oldSumTile, gZero);//初始化为0
    //初始化old_sum的tile
    //need
    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(i, 0);
        TLOAD(dataTile, gI);
        reducemax_col_kernel<tile_shapeData, tile_shapeMax><<<tile_shapeMax::ValidCol, tile_shapeMax::ValidRow, 1>>>(MaxTile.data(), dataTile.data(), oldMaxTile.data());
        oldMaxTile = MaxTile;
    }
    if constexpr (rmd_M > 0){
        auto gI = gIIter(Mb, 0);
        TLOAD(dataTile_col, gI);
        reducemax_col_kernel<tile_shapeData_col,tile_shapeMax><<<tile_shapeMax::ValidCol, tile_shapeMax::ValidRow, 1>>>(MaxTile.data(), dataTile_col.data(), oldMaxTile.data());
        oldMaxTile = MaxTile;
    }
    TSTORE(gO, MaxTile);
//    }
/*
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldMaxTile_row, 0);//初始化为0
//        TLOAD(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            reducemax_col_kernel<tile_shapeData_row,tile_shapeMax_row><<<tile_shapeMax_row::ValidCol, tile_shapeMax_row::ValidRow, 1>>>(MaxTile_row.data(), dataTile_row.data(), oldMaxTile_row.data());
            oldMaxTile_row = MaxTile_row;
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reducemax_col_kernel<tile_shapeData_cor,tile_shapeMax_row><<<tile_shapeMax_row::ValidCol, tile_shapeMax_row::ValidRow, 1>>>(MaxTile_row.data(), dataTile_cor.data(), oldMaxTile_row.data());
            oldMaxTile_row = MaxTile_row;
        }
        TSTORE(gO, MaxTile_row);
    }
*/
}

#endif
