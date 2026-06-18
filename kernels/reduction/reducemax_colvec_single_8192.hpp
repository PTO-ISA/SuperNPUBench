#ifndef REDUCEMAXCOLVEC_KERNEL_HPP
#define REDUCEMAXCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ==============================================
// ==============================================
template<typename tileSrc, typename tileTmpMax>
void __vec__ reducemax_col_kernel(
    typename tileTmpMax::TileDType __out__ new_max,
    const typename tileSrc::TileDType __in__ src,
    const typename tileTmpMax::TileDType __in__ old_max,
    const size_t tile_idx  
)
{
    size_t i = blkv_get_index_x();  

    __vbuf__ typename tileTmpMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileTmpMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);    

    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileTmpMax::ValidRow;j++){
        size_t old_max_idx =  i * tileTmpMax::ColStride + j * tileTmpMax::RowStride;       
        new_max_ptr[old_max_idx] = old_max_ptr[old_max_idx];          
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
        typename  tileSrc::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);    
        typename  tileSrc::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        typename  tileSrc::DType max_45 = blkv_max(src_ptr[src_idx_4], src_ptr[src_idx_5]);    
        typename  tileSrc::DType max_67 = blkv_max(src_ptr[src_idx_6], src_ptr[src_idx_7]);        
        typename  tileSrc::DType max_0123 = blkv_max(max_01, max_23); 
        typename  tileSrc::DType max_4567 = blkv_max(max_45, max_67);
        typename  tileSrc::DType max_all = blkv_max(max_0123, max_4567);   
        src_ptr[src_idx_0] = max_all;          
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
        typename tileSrc::DType tmp_max_01 = blkv_max(src_ptr[tmp_idx_0], src_ptr[tmp_idx_1]);
        typename tileSrc::DType tmp_max_23 = blkv_max(src_ptr[tmp_idx_2], src_ptr[tmp_idx_3]); 
        typename tileSrc::DType tmp_max_45 = blkv_max(src_ptr[tmp_idx_4], src_ptr[tmp_idx_5]); 
        typename tileSrc::DType tmp_max_67 = blkv_max(src_ptr[tmp_idx_6], src_ptr[tmp_idx_7]);  
        typename tileSrc::DType tmp_max_0123 = blkv_max(tmp_max_01, tmp_max_23); 
        typename tileSrc::DType tmp_max_4567 = blkv_max(tmp_max_45, tmp_max_67); 
        typename tileSrc::DType tmp_max_all = blkv_max(tmp_max_0123, tmp_max_4567);
        src_ptr[tmp_idx_0] = tmp_max_all;
    };



    size_t tmp_idx_l2_0 =  i * tileSrc::ColStride + 0*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_1 =  i * tileSrc::ColStride + 1*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_2 =  i * tileSrc::ColStride + 2*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_3 =  i * tileSrc::ColStride + 3*64 * tileSrc::RowStride;        
    size_t tmp_idx_l2_4 =  i * tileSrc::ColStride + 4*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_5 =  i * tileSrc::ColStride + 5*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_6 =  i * tileSrc::ColStride + 6*64 * tileSrc::RowStride;
    size_t tmp_idx_l2_7 =  i * tileSrc::ColStride + 7*64 * tileSrc::RowStride;      
    typename tileTmpMax::DType tmp_max_l2_01 = blkv_max(src_ptr[tmp_idx_l2_0], src_ptr[tmp_idx_l2_1]);
    typename tileTmpMax::DType tmp_max_l2_23 = blkv_max(src_ptr[tmp_idx_l2_2], src_ptr[tmp_idx_l2_3]);   
    typename tileTmpMax::DType tmp_max_l2_45 = blkv_max(src_ptr[tmp_idx_l2_4], src_ptr[tmp_idx_l2_5]);
    typename tileTmpMax::DType tmp_max_l2_67 = blkv_max(src_ptr[tmp_idx_l2_6], src_ptr[tmp_idx_l2_7]);  
    typename tileTmpMax::DType tmp_max_l2_0123 = blkv_max(tmp_max_l2_01, tmp_max_l2_23); 
    typename tileTmpMax::DType tmp_max_l2_4567 = blkv_max(tmp_max_l2_45, tmp_max_l2_67); 
    typename tileTmpMax::DType tmp_max_l2_all = blkv_max(tmp_max_l2_0123, tmp_max_l2_4567);          

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_max = upd_max + src_ptr[src_idx];              
    }
*/
//    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);        
//    new_max_ptr[i] = tmp_max_l2_all + old_max_ptr[i];  
//    new_max_ptr[i] = tmp_max_l2_all;  

    size_t  max_tile_idx = i * tileTmpMax::ColStride + tile_idx * tileTmpMax::RowStride;
    new_max_ptr[max_tile_idx] = tmp_max_l2_all;
}

template<typename tileTmpMax, typename tileMax>
void __vec__ reducemax_col_final_kernel(
    typename tileMax::TileDType __out__ new_max,
    const typename tileTmpMax::TileDType __in__ tmp_max
){
    size_t i = blkv_get_index_x();
    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileTmpMax::DType *tmp_max_ptr = blkv_get_tile_ptr(tmp_max);

    #pragma clang loop unroll(full) 
    for(size_t j=0;j<tileTmpMax::ValidRow;j+=8){
        size_t src_idx_0 =  i * tileTmpMax::ColStride + (j + 0) * tileTmpMax::RowStride;
        size_t src_idx_1 =  i * tileTmpMax::ColStride + (j + 1) * tileTmpMax::RowStride;
        size_t src_idx_2 =  i * tileTmpMax::ColStride + (j + 2) * tileTmpMax::RowStride;
        size_t src_idx_3 =  i * tileTmpMax::ColStride + (j + 3) * tileTmpMax::RowStride;        
        size_t src_idx_4 =  i * tileTmpMax::ColStride + (j + 4) * tileTmpMax::RowStride;
        size_t src_idx_5 =  i * tileTmpMax::ColStride + (j + 5) * tileTmpMax::RowStride;
        size_t src_idx_6 =  i * tileTmpMax::ColStride + (j + 6) * tileTmpMax::RowStride;
        size_t src_idx_7 =  i * tileTmpMax::ColStride + (j + 7) * tileTmpMax::RowStride;        
        typename  tileTmpMax::DType max_01 = blkv_max(tmp_max_ptr[src_idx_0], tmp_max_ptr[src_idx_1]);    
        typename  tileTmpMax::DType max_23 = blkv_max(tmp_max_ptr[src_idx_2], tmp_max_ptr[src_idx_3]); 
        typename  tileTmpMax::DType max_45 = blkv_max(tmp_max_ptr[src_idx_4], tmp_max_ptr[src_idx_5]);    
        typename  tileTmpMax::DType max_67 = blkv_max(tmp_max_ptr[src_idx_6], tmp_max_ptr[src_idx_7]);        
        typename  tileTmpMax::DType max_0123 = blkv_max(max_01, max_23); 
        typename  tileTmpMax::DType max_4567 = blkv_max(max_45, max_67);
        typename  tileTmpMax::DType max_all = blkv_max(max_0123, max_4567);   
        tmp_max_ptr[src_idx_0] = max_all;          
    }   

    size_t max_idx_0 = i * tileTmpMax::ColStride + 0*8 * tileTmpMax::RowStride;
    size_t max_idx_1 = i * tileTmpMax::ColStride + 1*8 * tileTmpMax::RowStride;
    new_max_ptr[i] = blkv_max(tmp_max_ptr[max_idx_0], tmp_max_ptr[max_idx_1]);
}


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
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //   
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //     
    using tile_shapeMax = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; // 
    using tile_shapeTmpMax = Tile<Location::Vec, dtype, 16, tN, BLayout::RowMajor>; //      


//    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // 
//    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //     
//    using tile_shapeMax_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; // 
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);   
    gm_shapeOut outGm(out_ptr); 

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;    
    tile_shapeMax MaxTile;
    tile_shapeTmpMax oldtmpMaxTile;
    tile_shapeTmpMax tmpMaxTile;
//    tile_shapeTmpMax_l2 tmpMaxTile_l2;

//    tile_shapeData_row dataTile_row;
//    tile_shapeData_cor dataTile_cor;    
//    tile_shapeMax_row MaxTile_row;
//    tile_shapeMax_row oldMaxTile_row;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn  gIIter(in_ptr);  
    itOut gOIter(out_ptr);

//    dtype zero = 0;

//    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpMaxTile, 0);//初始化为0
//    TEXPANDSCALAR(tmpMaxTile, 0);//初始化为0
//    TEXPANDSCALAR(tmpMaxTile_l2, 0);//初始化为0        
    for (size_t i = 0; i < Mb; ++i){
        auto gI = gIIter(i, 0);
        TCOPYIN(dataTile, gI);
        reducemax_col_kernel<tile_shapeData, tile_shapeTmpMax><<<tile_shapeTmpMax::ValidCol, 1, 1>>>(tmpMaxTile.data(), 
                                                                                                     dataTile.data(),
                                                                                                     oldtmpMaxTile.data(), 
                                                                                                     i);
        oldtmpMaxTile = tmpMaxTile;
    }
    reducemax_col_final_kernel<tile_shapeTmpMax, tile_shapeMax><<<tile_shapeMax::ValidCol, 1, 1>>>(MaxTile.data(), 
                                                                                                   tmpMaxTile.data());
    TCOPYOUT(gO, MaxTile);
}


#endif
