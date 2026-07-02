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
template<typename tileSrc, typename tileTmp>
void __vec__ reducesum_col_tmp(
    typename tileTmp::TileDType __out__ tmp_sum,
    const typename tileSrc::TileDType __in__ src    
)
{
    size_t i = blkv_get_index_x();  

    __vbuf__ typename tileTmp::DType *tmp_sum_ptr = blkv_get_tile_ptr(tmp_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
//    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   

    typename tileTmp::DType upd_tmp_sum = 0;
   
    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileSrc::Rows;j+=8){//非valid处也参与计算补0，能凑出8元树形累加出来
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;
        size_t src_idx_4 =  i * tileSrc::ColStride + (j + 4) * tileSrc::RowStride;
        size_t src_idx_5 =  i * tileSrc::ColStride + (j + 5) * tileSrc::RowStride;
        size_t src_idx_6 =  i * tileSrc::ColStride + (j + 6) * tileSrc::RowStride;
        size_t src_idx_7 =  i * tileSrc::ColStride + (j + 7) * tileSrc::RowStride;        
        typename tileTmp::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];    
        typename tileTmp::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename tileTmp::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];    
        typename tileTmp::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];        
        typename tileTmp::DType sum_0123 = sum_01 + sum_23; 
        typename tileTmp::DType sum_4567 = sum_45 + sum_67;
        typename tileTmp::DType sum_tmp = sum_0123 + sum_4567;         
        upd_tmp_sum = upd_tmp_sum + sum_tmp;              
    }

    tmp_sum_ptr[i] = upd_tmp_sum;   
}

template<typename tileTmp, typename tileSum>
void __vec__ reducesum_col_final(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileTmp::TileDType __in__ src, 
    const typename tileSum::TileDType __in__ old_sum   
)
{
    size_t i = blkv_get_index_x();  

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileTmp::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   


    typename tileSum::DType upd_sum = old_sum_ptr[i];
   

    size_t src_idx_0 =  i * tileSum::ColStride + 0 * tileSum::ValidCol;
    size_t src_idx_1 =  i * tileSum::ColStride + 1 * tileSum::ValidCol;
    size_t src_idx_2 =  i * tileSum::ColStride + 2 * tileSum::ValidCol;
    size_t src_idx_3 =  i * tileSum::ColStride + 3 * tileSum::ValidCol;  
    size_t src_idx_4 =  i * tileSum::ColStride + 4 * tileSum::ValidCol;
    size_t src_idx_5 =  i * tileSum::ColStride + 5 * tileSum::ValidCol;
    size_t src_idx_6 =  i * tileSum::ColStride + 6 * tileSum::ValidCol;
    size_t src_idx_7 =  i * tileSum::ColStride + 7 * tileSum::ValidCol;       
    typename tileSum::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];    
    typename tileSum::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];    
    typename tileSum::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];    
    typename tileSum::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];        
    typename tileSum::DType sum_0123 = sum_01 + sum_23; 
    typename tileSum::DType sum_4567 = sum_45 + sum_67;      
    typename tileSum::DType sum_all = sum_0123 + sum_4567; 
              
//        upd_sum = upd_sum + sum_tmp;              

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_sum = upd_sum + src_ptr[src_idx];              
    }
*/        
//    new_sum_ptr[i] = upd_sum; 
    new_sum_ptr[i] = sum_all + upd_sum;   
}





template<typename dtype, int gIM, int gIN, int tM, int tN, int tM_VLD>
void reducesum_colsum_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,    
    dtype *out_ptr
) 
{

//    const int Mb = (gIM/8) / tM;  

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM/8, gIN*8>>;     // 
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;    
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM/8, tN*8, BLayout::RowMajor, tM_VLD/8, tN*8>; //
//    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //
    using tile_shapeTmp = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor>; //      
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor, 1, tN>; // 



//    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // 
//    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //     
//    using tile_shapeSum_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; // 
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);   
//    gm_shapeOut ZeroGm(inzero_ptr); 
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;
//    tile_shapeData_col dataTile_col;
    tile_shapeTmp TmpTile;        
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;

//    tile_shapeData_row dataTile_row;
//    tile_shapeData_cor dataTile_cor;    
//    tile_shapeSum_row SumTile_row;
//    tile_shapeSum_row oldSumTile_row;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;      
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeSum>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);    
    itOut gOIter(out_ptr);


    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldSumTile, 0);//初始化为0
    auto gI = gIIter(0, 0);
    TCOPYIN(dataTile, gI);//TLOAD应补0，目前gfrun默认补0，需要接口去弄
    reducesum_col_tmp<tile_shapeData, tile_shapeTmp><<<tile_shapeTmp::ValidCol, tile_shapeTmp::ValidRow, 1>>>(TmpTile.data(), dataTile.data());
    reducesum_col_final<tile_shapeTmp, tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), TmpTile.data(), oldSumTile.data());
    oldSumTile = SumTile;
    TCOPYOUT(gO, SumTile);
}

#endif
