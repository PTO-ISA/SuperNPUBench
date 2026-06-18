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
template<typename tileSrc, typename tileTmpSum>
void __vec__ reducesum_col_kernel(
    typename tileTmpSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileTmpSum::TileDType __in__ old_sum,
    const size_t tile_idx  
)
{
    size_t i = blkv_get_index_x();  

    __vbuf__ typename tileTmpSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileTmpSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);    

    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileTmpSum::ValidRow;j++){
        size_t old_sum_idx =  i * tileTmpSum::ColStride + j * tileTmpSum::RowStride;       
        new_sum_ptr[old_sum_idx] = old_sum_ptr[old_sum_idx];          
    }
    
    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileSrc::ValidRow;j+=8){
        size_t src_idx_0 =  i * tileSrc::ColStride + (j + 0) * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;        
        size_t src_idx_4 =  i * tileSrc::ColStride + (j + 4) * tileSrc::RowStride;
        size_t src_idx_5 =  i * tileSrc::ColStride + (j + 5) * tileSrc::RowStride;
        size_t src_idx_6 =  i * tileSrc::ColStride + (j + 6) * tileSrc::RowStride;
        size_t src_idx_7 =  i * tileSrc::ColStride + (j + 7) * tileSrc::RowStride;
        typename  tileSrc::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];    
        typename  tileSrc::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename  tileSrc::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];    
        typename  tileSrc::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];        
        typename  tileSrc::DType sum_0123 = sum_01 + sum_23;
        typename  tileSrc::DType sum_4567 = sum_45 + sum_67;
        typename  tileSrc::DType sum_all = sum_0123 + sum_4567;   
        src_ptr[src_idx_0] = sum_all;          
    }

    #pragma clang loop unroll(full)
    for(size_t j=0; j<tileSrc::ValidRow; j+=64){
        size_t tmp_idx_0 =  i * tileSrc::ColStride + (j + 0*8) * tileSrc::RowStride;
        size_t tmp_idx_1 =  i * tileSrc::ColStride + (j + 1*8) * tileSrc::RowStride;
        size_t tmp_idx_2 =  i * tileSrc::ColStride + (j + 2*8) * tileSrc::RowStride;
        size_t tmp_idx_3 =  i * tileSrc::ColStride + (j + 3*8) * tileSrc::RowStride;        
        size_t tmp_idx_4 =  i * tileSrc::ColStride + (j + 4*8) * tileSrc::RowStride;
        size_t tmp_idx_5 =  i * tileSrc::ColStride + (j + 5*8) * tileSrc::RowStride;
        size_t tmp_idx_6 =  i * tileSrc::ColStride + (j + 6*8) * tileSrc::RowStride;
        size_t tmp_idx_7 =  i * tileSrc::ColStride + (j + 7*8) * tileSrc::RowStride;  
        typename tileSrc::DType tmp_sum_01 = src_ptr[tmp_idx_0]+ src_ptr[tmp_idx_1];
        typename tileSrc::DType tmp_sum_23 = src_ptr[tmp_idx_2]+ src_ptr[tmp_idx_3]; 
        typename tileSrc::DType tmp_sum_45 = src_ptr[tmp_idx_4]+ src_ptr[tmp_idx_5]; 
        typename tileSrc::DType tmp_sum_67 = src_ptr[tmp_idx_6]+ src_ptr[tmp_idx_7];  
        typename tileSrc::DType tmp_sum_0123 = tmp_sum_01 + tmp_sum_23; 
        typename tileSrc::DType tmp_sum_4567 = tmp_sum_45 + tmp_sum_67; 
        typename tileSrc::DType tmp_sum_all = tmp_sum_0123 + tmp_sum_4567;
        src_ptr[tmp_idx_0] = tmp_sum_all;
    };



    size_t tmp_idx_l2_0 =  i * tileSrc::ColStride + 0*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_1 =  i * tileSrc::ColStride + 1*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_2 =  i * tileSrc::ColStride + 2*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_3 =  i * tileSrc::ColStride + 3*64 * tileSrc::RowStride;        
    size_t tmp_idx_l2_4 =  i * tileSrc::ColStride + 4*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_5 =  i * tileSrc::ColStride + 5*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_6 =  i * tileSrc::ColStride + 6*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_7 =  i * tileSrc::ColStride + 7*64 * tileSrc::RowStride;      
    typename tileTmpSum::DType tmp_sum_l2_01 = src_ptr[tmp_idx_l2_0] + src_ptr[tmp_idx_l2_1];
    typename tileTmpSum::DType tmp_sum_l2_23 = src_ptr[tmp_idx_l2_2] + src_ptr[tmp_idx_l2_3];   
    typename tileTmpSum::DType tmp_sum_l2_45 = src_ptr[tmp_idx_l2_4] + src_ptr[tmp_idx_l2_5];
    typename tileTmpSum::DType tmp_sum_l2_67 = src_ptr[tmp_idx_l2_6] + src_ptr[tmp_idx_l2_7];  
    typename tileTmpSum::DType tmp_sum_l2_0123 = tmp_sum_l2_01 + tmp_sum_l2_23; 
    typename tileTmpSum::DType tmp_sum_l2_4567 = tmp_sum_l2_45 + tmp_sum_l2_67; 
    typename tileTmpSum::DType tmp_sum_l2_all = tmp_sum_l2_0123 + tmp_sum_l2_4567;          

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_sum = upd_sum + src_ptr[src_idx];              
    }
*/
//    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);        
//    new_sum_ptr[i] = tmp_sum_l2_all + old_sum_ptr[i];  
//    new_sum_ptr[i] = tmp_sum_l2_all;  

    size_t  sum_tile_idx = i * tileTmpSum::ColStride + tile_idx * tileTmpSum::RowStride;
    new_sum_ptr[sum_tile_idx] = tmp_sum_l2_all;
}

template<typename tileTmpSum, typename tileSum>
void __vec__ reducesum_col_final_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileTmpSum::TileDType __in__ tmp_sum
){
    size_t i = blkv_get_index_x();
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileTmpSum::DType *tmp_sum_ptr = blkv_get_tile_ptr(tmp_sum);

    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileTmpSum::ValidRow;j+=8){
        size_t src_idx_0 =  i * tileTmpSum::ColStride + (j + 0) * tileTmpSum::RowStride;
        size_t src_idx_1 =  i * tileTmpSum::ColStride + (j + 1) * tileTmpSum::RowStride;
        size_t src_idx_2 =  i * tileTmpSum::ColStride + (j + 2) * tileTmpSum::RowStride;
        size_t src_idx_3 =  i * tileTmpSum::ColStride + (j + 3) * tileTmpSum::RowStride;        
        size_t src_idx_4 =  i * tileTmpSum::ColStride + (j + 4) * tileTmpSum::RowStride;
        size_t src_idx_5 =  i * tileTmpSum::ColStride + (j + 5) * tileTmpSum::RowStride;
        size_t src_idx_6 =  i * tileTmpSum::ColStride + (j + 6) * tileTmpSum::RowStride;
        size_t src_idx_7 =  i * tileTmpSum::ColStride + (j + 7) * tileTmpSum::RowStride;        
        typename  tileTmpSum::DType sum_01 = tmp_sum_ptr[src_idx_0] + tmp_sum_ptr[src_idx_1];    
        typename  tileTmpSum::DType sum_23 = tmp_sum_ptr[src_idx_2] + tmp_sum_ptr[src_idx_3];
        typename  tileTmpSum::DType sum_45 = tmp_sum_ptr[src_idx_4] + tmp_sum_ptr[src_idx_5];    
        typename  tileTmpSum::DType sum_67 = tmp_sum_ptr[src_idx_6] + tmp_sum_ptr[src_idx_7];        
        typename  tileTmpSum::DType sum_0123 = sum_01 + sum_23; 
        typename  tileTmpSum::DType sum_4567 = sum_45 + sum_67;
        typename  tileTmpSum::DType sum_all = sum_0123 + sum_4567;   
        tmp_sum_ptr[src_idx_0] = sum_all;          
    }   

    size_t sum_idx_0 = i * tileTmpSum::ColStride + 0*8 * tileTmpSum::RowStride;
    size_t sum_idx_1 = i * tileTmpSum::ColStride + 1*8 * tileTmpSum::RowStride;
    new_sum_ptr[i] = tmp_sum_ptr[sum_idx_0] + tmp_sum_ptr[sum_idx_1];
}


template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducesum_colsum_rand(
    dtype *in_ptr,  
    dtype *out_ptr
) 
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;    

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //   
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //     
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; // 
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, 16, tN, BLayout::RowMajor>; // 
//    using tile_shapeTmpSum_l2 = Tile<Location::Vec, dtype, tM/64, tN, BLayout::RowMajor>; //     


//    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // 
//    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //     
//    using tile_shapeSum_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; // 
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);   
    gm_shapeOut outGm(out_ptr); 

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;    
    tile_shapeSum SumTile;
    tile_shapeTmpSum oldtmpSumTile;
    tile_shapeTmpSum tmpSumTile;
//    tile_shapeTmpSum_l2 tmpSumTile_l2;

//    tile_shapeData_row dataTile_row;
//    tile_shapeData_cor dataTile_cor;    
//    tile_shapeSum_row SumTile_row;
//    tile_shapeSum_row oldSumTile_row;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);  
    itOut gOIter(out_ptr);

//    dtype zero = 0;

//    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpSumTile, 0);//初始化为0
//    TEXPANDSCALAR(tmpSumTile, 0);//初始化为0
//    TEXPANDSCALAR(tmpSumTile_l2, 0);//初始化为0        
    for (size_t i = 0; i < Mb; ++i){
        auto gI = gIIter(i, 0);
        TCOPYIN(dataTile, gI);
        reducesum_col_kernel<tile_shapeData, tile_shapeTmpSum><<<tile_shapeTmpSum::ValidCol, 1, 1>>>(tmpSumTile.data(), 
                                                                                                     dataTile.data(),
                                                                                                     oldtmpSumTile.data(), 
                                                                                                     i);
        oldtmpSumTile = tmpSumTile;
    }
    reducesum_col_final_kernel<tile_shapeTmpSum, tile_shapeSum><<<tile_shapeSum::ValidCol, 1, 1>>>(SumTile.data(), 
                                                                                                   tmpSumTile.data());
    TCOPYOUT(gO, SumTile);
}
/*
    if constexpr (rmd_M > 0){   
        auto gI = gIIter(Mb, 0);
        TCOPYIN(dataTile_col, gI);
        reducesum_col_kernel<tile_shapeData_col,tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile_col.data(), oldSumTile.data());
        oldSumTile = SumTile;
    }
    */
//}
//    }
    /*
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);         
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldSumTile_row, 0);//初始化为0        
//        TCOPYIN(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {   
            auto gI = gIIter(i, Nb);
            TCOPYIN(dataTile_row, gI);
            reducesum_col_kernel<tile_shapeData_row,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_row.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        if constexpr (rmd_M > 0){   
            auto gI = gIIter(Mb, Nb);
            TCOPYIN(dataTile_cor, gI);
            reducesum_col_kernel<tile_shapeData_cor,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_cor.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        TCOPYOUT(gO, SumTile_row);
    }
    */


#endif
