#ifndef REDUCEMAXTROWMAX_KERNEL_HPP
#define REDUCEMAXTROWMAX_KERNEL_HPP

#ifndef __vbuf__
#define __vbuf__
#endif

#include <common/pto_tileop.hpp>

using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>

template<typename tileSrc, typename tileSrcCol, typename tileTmpMax>
void __vec__ reducemax_row_kernel(
    typename tileTmpMax::TileDType __out__ new_max,
    const typename tileSrc::TileDType __in__ src,
    const typename tileSrcCol::TileDType __in__ src_col,
    const typename tileTmpMax::TileDType __in__ old_max,
    const size_t tile_idx    
)
{

    size_t j = blkv_get_index_x();  
    size_t z = blkv_get_index_y();     
    size_t stride_src = z * (tileSrc::ValidCol/4) * tileSrc::ColStride;
    size_t stride_src_col = z * (tileSrcCol::ValidCol/4) * tileSrcCol::ColStride;    
  
    __vbuf__ typename tileTmpMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSrc::DType *src_col_ptr = blkv_get_tile_ptr(src_col);    
    __vbuf__ typename tileTmpMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);   

/*
    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpMax::ValidCol/4;i++){
        size_t old_max_idx =  z * tileTmpMax::ValidCol/4 * tileTmpMax::ColStride + i * tileTmpMax::ColStride + j * tileTmpMax::RowStride;       
        new_max_ptr[old_max_idx] = old_max_ptr[old_max_idx];          
    }    
*/

    #pragma clang loop unroll(full)
    for(size_t i=0;i<tileSrc::ValidCol/4;i+=8){
        size_t src_idx_0 =  stride_src + (i+0) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  stride_src + (i+1) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_2 =  stride_src + (i+2) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_3 =  stride_src + (i+3) * tileSrc::ColStride + j * tileSrc::RowStride;        
        size_t src_idx_4 =  stride_src + (i+4) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_5 =  stride_src + (i+5) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_6 =  stride_src + (i+6) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_7 =  stride_src + (i+7) * tileSrc::ColStride + j * tileSrc::RowStride; 

        typename tileSrc::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileSrc::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);   
        typename tileSrc::DType max_45 = blkv_max(src_ptr[src_idx_4], src_ptr[src_idx_5]);  
        typename tileSrc::DType max_67 = blkv_max(src_ptr[src_idx_6], src_ptr[src_idx_7]);    

        typename tileSrc::DType max_0123 = blkv_max(max_01, max_23);
        typename tileSrc::DType max_4567 = blkv_max(max_45, max_67);        

        typename tileSrc::DType max_all = blkv_max(max_0123, max_4567);

        size_t src_col_idx_0 = stride_src_col + (i/8) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        src_col_ptr[src_col_idx_0] = max_all;         
    }        


    #pragma clang loop unroll(full)
    for(size_t i=0; i<tileSrcCol::ValidCol/4; i+=8){
        size_t tmp_idx_0 =  stride_src_col + (i+0) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_1 =  stride_src_col + (i+1) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_2 =  stride_src_col + (i+2) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_3 =  stride_src_col + (i+3) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;        
        size_t tmp_idx_4 =  stride_src_col + (i+4) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_5 =  stride_src_col + (i+5) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_6 =  stride_src_col + (i+6) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        size_t tmp_idx_7 =  stride_src_col + (i+7) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;  
        typename tileSrcCol::DType tmp_max_01 = blkv_max(src_col_ptr[tmp_idx_0], src_col_ptr[tmp_idx_1]);
        typename tileSrcCol::DType tmp_max_23 = blkv_max(src_col_ptr[tmp_idx_2], src_col_ptr[tmp_idx_3]); 
        typename tileSrcCol::DType tmp_max_45 = blkv_max(src_col_ptr[tmp_idx_4], src_col_ptr[tmp_idx_5]); 
        typename tileSrcCol::DType tmp_max_67 = blkv_max(src_col_ptr[tmp_idx_6], src_col_ptr[tmp_idx_7]);  
        typename tileSrcCol::DType tmp_max_0123 = blkv_max(tmp_max_01, tmp_max_23); 
        typename tileSrcCol::DType tmp_max_4567 = blkv_max(tmp_max_45, tmp_max_67); 
        typename tileSrcCol::DType tmp_max_all = blkv_max(tmp_max_0123, tmp_max_4567);
        src_col_ptr[tmp_idx_0] = tmp_max_all;
    }



    size_t stride = 8;
    size_t iternum = __builtin_ctz(tileSrcCol::ValidCol/4) - 3;

    #pragma clang loop unroll(full) 
    for(size_t k=0; k<iternum; k++){
        //#pragma clang loop unroll(full) 
        for(size_t i=0; i<tileSrcCol::ValidCol/4; i+=(stride*2)){
            size_t src_idx_0 =  stride_src_col + (i + 0*stride) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
            size_t src_idx_1 =  stride_src_col + (i + 1*stride) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
            typename  tileSrcCol::DType max_01 = blkv_max(src_col_ptr[src_idx_0], src_col_ptr[src_idx_1]);           
            src_col_ptr[src_idx_0] = max_01;          
        }
        stride = stride*2;
    }


    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpMax::ValidCol/4;i++){
        size_t old_max_idx =  z * tileTmpMax::ValidCol/4 * tileTmpMax::ColStride + i * tileTmpMax::ColStride + j * tileTmpMax::RowStride;       
        new_max_ptr[old_max_idx] = old_max_ptr[old_max_idx];          
    }    


    size_t src_max_idx = stride_src_col + j * tileSrcCol::RowStride;
    size_t max_tile_idx = z * tileTmpMax::ValidCol/4 * tileTmpMax::ColStride + tile_idx * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
    new_max_ptr[max_tile_idx] = src_col_ptr[src_max_idx];  
}


template<typename tileTmpMax, typename tileMax>
void __vec__ reducemax_row_final_kernel(
    typename tileMax::TileDType __out__ new_max,
    const typename tileTmpMax::TileDType __in__ tmp_max
){
    size_t j = blkv_get_index_x();
    size_t idx = j * tileMax::RowStride;

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileTmpMax::DType *tmp_max_ptr = blkv_get_tile_ptr(tmp_max);

    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpMax::Cols;i+=8){
        size_t src_idx_0 =  (i+0) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_1 =  (i+1) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_2 =  (i+2) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_3 =  (i+3) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;        
        size_t src_idx_4 =  (i+4) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_5 =  (i+5) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_6 =  (i+6) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
        size_t src_idx_7 =  (i+7) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;        
        typename  tileTmpMax::DType max_01 = blkv_max(tmp_max_ptr[src_idx_0], tmp_max_ptr[src_idx_1]);    
        typename  tileTmpMax::DType max_23 = blkv_max(tmp_max_ptr[src_idx_2], tmp_max_ptr[src_idx_3]);
        typename  tileTmpMax::DType max_45 = blkv_max(tmp_max_ptr[src_idx_4], tmp_max_ptr[src_idx_5]);    
        typename  tileTmpMax::DType max_67 = blkv_max(tmp_max_ptr[src_idx_6], tmp_max_ptr[src_idx_7]);        
        typename  tileTmpMax::DType max_0123 = blkv_max(max_01, max_23); 
        typename  tileTmpMax::DType max_4567 = blkv_max(max_45, max_67);
        typename  tileTmpMax::DType max_all = blkv_max(max_0123, max_4567);   
        tmp_max_ptr[src_idx_0] = max_all;          
    }   

    size_t stride = 8;
    size_t iternum = __builtin_ctz(tileTmpMax::Cols) - 3;    
    #pragma clang loop unroll(full) 
    for(size_t k=0;k<iternum;k++){
        #pragma clang loop unroll(full) 
        for(size_t i=0;i<tileTmpMax::Cols;i+=(stride*2)){
            size_t src_idx_0 =  (i + 0*stride) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
            size_t src_idx_1 =  (i + 1*stride) * tileTmpMax::ColStride + j * tileTmpMax::RowStride;
            typename  tileTmpMax::DType max_01 = blkv_max(tmp_max_ptr[src_idx_0], tmp_max_ptr[src_idx_1]);           
            tmp_max_ptr[src_idx_0] = max_01;          
        }
        stride = stride*2;
    }    

    size_t max_idx = j * tileTmpMax::RowStride;
    new_max_ptr[idx] = tmp_max_ptr[max_idx];
}


template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reducemax_row_rand(
    dtype *in_ptr,
    dtype *out_ptr
) 
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;    

    const int rmd_M = gIM % tM; // todo 尾块怎么处理？
    const int rmd_N = gIN % tN; // todo 尾块怎么处理？    


    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //将gm中的Tensor先声明为一维数据 
//    using gm_shapeMax = global_tensor<dtype, RowMajor<gIM, gIN>>;    
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, 1>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeDataCol = Tile<Location::Vec, dtype, tM, tN/8, BLayout::ColMajor>; // todo 尾块怎么处理？是否要作为参数写在这    
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec
    using tile_shapeTmpMax = Tile<Location::Vec, dtype, tM, 64, BLayout::ColMajor, tM, Nb*4>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeMax olcMaxGm(old_max_ptr);    

    tile_shapeData dataTile;   
    tile_shapeDataCol dataTile_col;                        
    tile_shapeMax MaxTile;
    tile_shapeTmpMax oldtmpMaxTile;
    tile_shapeTmpMax tmpMaxTile;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);


    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpMaxTile, 0);//初始化为0  
    TEXPANDSCALAR(dataTile_col, 0);//初始化为0     
    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);                
        TCOPYIN(dataTile, gI);    
        reducemax_row_kernel<tile_shapeData, tile_shapeDataCol, tile_shapeTmpMax><<<tile_shapeTmpMax::ValidRow, 4, 1>>>(tmpMaxTile.data(), 
                                                                                                                        dataTile.data(), 
                                                                                                                        dataTile_col.data(), 
                                                                                                                        oldtmpMaxTile.data(),
                                                                                                                        i);
        oldtmpMaxTile = tmpMaxTile;
    }
    reducemax_row_final_kernel<tile_shapeTmpMax, tile_shapeMax><<<tile_shapeTmpMax::ValidRow, 1, 1>>>(MaxTile.data(), 
                                                                                                      tmpMaxTile.data());     
    TCOPYOUT(gO, MaxTile);
}

#endif

