#ifndef REDUCESUMTROWSUM_KERNEL_HPP
#define REDUCESUMTROWSUM_KERNEL_HPP

#ifndef __vbuf__
#define __vbuf__
#endif

#include <common/pto_tileop.hpp>

using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>

template<typename tileSrc, typename tileTmpSum>
void __vec__ reducesum_row_kernel(
    typename tileTmpSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileTmpSum::TileDType __in__ old_sum,
    const size_t tile_idx    
)
{

    size_t j = blkv_get_index_x();  
  
    __vbuf__ typename tileTmpSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileTmpSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   

    #pragma clang loop unroll(full)
    for(int i=0;i<tileSrc::ValidCol;i+=8){
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  (i+1) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_2 =  (i+2) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_3 =  (i+3) * tileSrc::ColStride + j * tileSrc::RowStride;        
        size_t src_idx_4 =  (i+4) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_5 =  (i+5) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_6 =  (i+6) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_7 =  (i+7) * tileSrc::ColStride + j * tileSrc::RowStride; 

        typename tileSrc::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileSrc::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];   
        typename tileSrc::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];  
        typename tileSrc::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];    

        typename tileSrc::DType sum_0123 = sum_01 + sum_23;
        typename tileSrc::DType sum_4567 = sum_45 + sum_67;        

        typename tileSrc::DType sum_all = sum_0123 + sum_4567;
        src_ptr[src_idx_0] = sum_all;         
    }        


    #pragma clang loop unroll(full)
    for(int i=0; i<tileSrc::ValidCol; i+=64){
        size_t tmp_idx_0 =  (i+0*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_1 =  (i+1*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_2 =  (i+2*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_3 =  (i+3*8) * tileSrc::ColStride + j * tileSrc::RowStride;        
        size_t tmp_idx_4 =  (i+4*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_5 =  (i+5*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_6 =  (i+6*8) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t tmp_idx_7 =  (i+7*8) * tileSrc::ColStride + j * tileSrc::RowStride;  
        typename tileSrc::DType tmp_sum_01 = src_ptr[tmp_idx_0]+ src_ptr[tmp_idx_1];
        typename tileSrc::DType tmp_sum_23 = src_ptr[tmp_idx_2]+ src_ptr[tmp_idx_3]; 
        typename tileSrc::DType tmp_sum_45 = src_ptr[tmp_idx_4]+ src_ptr[tmp_idx_5]; 
        typename tileSrc::DType tmp_sum_67 = src_ptr[tmp_idx_6]+ src_ptr[tmp_idx_7];  
        typename tileSrc::DType tmp_sum_0123 = tmp_sum_01 + tmp_sum_23; 
        typename tileSrc::DType tmp_sum_4567 = tmp_sum_45 + tmp_sum_67; 
        typename tileSrc::DType tmp_sum_all = tmp_sum_0123 + tmp_sum_4567;
        src_ptr[tmp_idx_0] = tmp_sum_all;
    }

    size_t stride = 64;
    size_t iternum = __builtin_ctz(tileSrc::ValidCol) - 6;

//    #pragma clang loop unroll(full) 
    for(size_t k=0; k<iternum; k++){
        //#pragma clang loop unroll(full) 
        for(size_t i=0; i<tileSrc::ValidCol; i+=(stride*2)){
            size_t src_idx_0 =  (i + 0*stride) * tileSrc::ColStride + j * tileSrc::RowStride;
            size_t src_idx_1 =  (i + 1*stride) * tileSrc::ColStride + j * tileSrc::RowStride;
            typename  tileSrc::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];           
            src_ptr[src_idx_0] = sum_01;          
        }
        stride = stride*2;
    }


    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpSum::ValidCol;i++){
        size_t old_sum_idx =  i * tileTmpSum::ColStride + j * tileTmpSum::RowStride;       
        new_sum_ptr[old_sum_idx] = old_sum_ptr[old_sum_idx];          
    }    


    size_t src_sum_idx = j * tileSrc::RowStride;
    size_t  sum_tile_idx = tile_idx * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
    new_sum_ptr[sum_tile_idx] = src_ptr[src_sum_idx];  
}


template<typename tileTmpSum, typename tileSum>
void __vec__ reducesum_row_final_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileTmpSum::TileDType __in__ tmp_sum
){
    size_t j = blkv_get_index_x();
    size_t idx = j * tileSum::RowStride;

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileTmpSum::DType *tmp_sum_ptr = blkv_get_tile_ptr(tmp_sum);

    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpSum::ValidCol;i+=8){
        size_t src_idx_0 =  (i+0) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_1 =  (i+1) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_2 =  (i+2) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_3 =  (i+3) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;        
        size_t src_idx_4 =  (i+4) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_5 =  (i+5) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_6 =  (i+6) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
        size_t src_idx_7 =  (i+7) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;        
        typename  tileTmpSum::DType sum_01 = tmp_sum_ptr[src_idx_0] + tmp_sum_ptr[src_idx_1];    
        typename  tileTmpSum::DType sum_23 = tmp_sum_ptr[src_idx_2] + tmp_sum_ptr[src_idx_3];
        typename  tileTmpSum::DType sum_45 = tmp_sum_ptr[src_idx_4] + tmp_sum_ptr[src_idx_5];    
        typename  tileTmpSum::DType sum_67 = tmp_sum_ptr[src_idx_6] + tmp_sum_ptr[src_idx_7];        
        typename  tileTmpSum::DType sum_0123 = sum_01 + sum_23; 
        typename  tileTmpSum::DType sum_4567 = sum_45 + sum_67;
        typename  tileTmpSum::DType sum_all = sum_0123 + sum_4567;   
        tmp_sum_ptr[src_idx_0] = sum_all;          
    }   

    size_t stride = 8;
    size_t iternum = __builtin_ctz(tileTmpSum::ValidCol) - 3;    
    #pragma clang loop unroll(full) 
    for(size_t k=0;k<iternum;k++){
        #pragma clang loop unroll(full) 
        for(size_t i=0;i<tileTmpSum::ValidCol;i+=(stride*2)){
            size_t src_idx_0 =  (i + 0*stride) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
            size_t src_idx_1 =  (i + 1*stride) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
            typename  tileTmpSum::DType sum_01 = tmp_sum_ptr[src_idx_0] + tmp_sum_ptr[src_idx_1];           
            tmp_sum_ptr[src_idx_0] = sum_01;          
        }
        stride = stride*2;
    }    

    size_t sum_idx = j * tileTmpSum::RowStride;
    new_sum_ptr[idx] = tmp_sum_ptr[sum_idx];
}


template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reducesum_trowsum_rand(
    dtype *in_ptr,
    dtype *out_ptr
) 
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;    

    const int rmd_M = gIM % tM; // todo 尾块怎么处理？
    const int rmd_N = gIN % tN; // todo 尾块怎么处理？    


    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //将gm中的Tensor先声明为一维数据 
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;    
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, 1>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeSum = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, tM, Nb>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;                      
    tile_shapeSum SumTile;
    tile_shapeTmpSum oldtmpSumTile;
    tile_shapeTmpSum tmpSumTile;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);


//    for (int j = 0; j < Mb; ++j) {
    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpSumTile, 0);//初始化为0      
    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);   
//       printf("before copy in , %d\n", i);                
        TCOPYIN(dataTile, gI);    
        reducesum_row_kernel<tile_shapeData, tile_shapeTmpSum><<<tile_shapeTmpSum::ValidRow, 1, 1>>>(tmpSumTile.data(), 
                                                                                                     dataTile.data(), 
                                                                                                     oldtmpSumTile.data(),
                                                                                                     i);
//      reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//      printf("kernel , %d\n", i);
        oldtmpSumTile = tmpSumTile;
    }
    reducesum_row_final_kernel<tile_shapeTmpSum, tile_shapeSum><<<tile_shapeTmpSum::ValidRow, 1, 1>>>(SumTile.data(), 
                                                                                                      tmpSumTile.data());     
    TCOPYOUT(gO, SumTile);
}
//}

#endif
