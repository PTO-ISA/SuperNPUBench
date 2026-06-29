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

template<typename tileData, typename tileSum>
void __vec__ cumsum_row_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileData::TileDType __out__ out,    
    const typename tileData::TileDType __in__ src,
    const typename tileSum::TileDType __in__ old_sum    
)
{
//    size_t i = blkv_get_index_x();  
    size_t j = blkv_get_index_y();  
    size_t sum_idx = j * tileSum::RowStride;    

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileData::DType *out_ptr = blkv_get_tile_ptr(out);    
    __vbuf__ typename tileData::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   


    typename tileSum::DType upd_sum = old_sum_ptr[sum_idx];
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
    for(size_t i=0;i<tileData::ValidCol;i++){
        size_t idx =  i * tileData::ColStride + j * tileData::RowStride;
        typename tileData::DType sum_out = upd_sum + src_ptr[idx];
        upd_sum = sum_out;   
        out_ptr[idx] = static_cast<typename tileData::DType>(sum_out);           
    }    
    new_sum_ptr[sum_idx] = upd_sum;    
}



template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void cumsum_row_rand(
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
    using gm_shapeOut = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeSum = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>; // todo 尾块怎么处理？是否要作为参数写在这   
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeSum_col =  Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, rmd_M, 1>;      


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;                
    tile_shapeData_row dataTile_row;
    tile_shapeData_col dataTile_col;
    tile_shapeData_cor dataTile_cor;    

    tile_shapeData OutTile;                
    tile_shapeData_row OutTile_row;
    tile_shapeData_col OutTile_col;
    tile_shapeData_cor OutTile_cor;      
    
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;
    tile_shapeSum_col SumTile_col;
    tile_shapeSum_col oldSumTile_col;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

//    printf("tile_shapeSum::ValidCol = %d\n",  tile_shapeSum::ValidCol);
//    printf("tile_shapeSum::ValidRow = %d\n",  tile_shapeSum::ValidRow);    

    for (int j = 0; j < Mb; ++j) {
//        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldSumTile, 0);//初始化为0
        //初始化old_sum的tile      
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i); 
            auto gO = gOIter(j, i);                               
            TCOPYIN(dataTile, gI);    
            cumsum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), OutTile.data(), dataTile.data(), oldSumTile.data());
//            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
            oldSumTile = SumTile;
            TCOPYOUT(gO, OutTile);            
        }
//        printf("end for%d\n",j);
        //for row corner
        if constexpr (rmd_N > 0){
            auto gI = gIIter(j, Nb);
            auto gO = gOIter(j, Nb);
            TCOPYIN(dataTile_row, gI);
            cumsum_row_kernel<tile_shapeData_row, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), OutTile_row.data(), dataTile_row.data(), oldSumTile.data());            
//            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
            oldSumTile = SumTile;
            TCOPYOUT(gO, OutTile_row);            
        }
    }
    //for col cor
    if constexpr (rmd_M > 0){
        TEXPANDSCALAR(oldSumTile_col, 0);//初始化为0
        //初始化old_sum的tile      
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);   
            auto gO = gOIter(Mb, i);   
            TCOPYIN(dataTile_col, gI);                  
            cumsum_row_kernel<tile_shapeData_col, tile_shapeSum_col><<<1, tile_shapeSum_col::ValidRow, 1>>>(SumTile_col.data(), OutTile_col.data(), dataTile_col.data(), oldSumTile_col.data());
            oldSumTile_col = SumTile_col;
            TCOPYOUT(gO, OutTile_col);
        }
        if constexpr (rmd_N > 0){
            auto gI = gIIter(Mb, Nb);
            auto gO = gOIter(Mb, Nb);
            TCOPYIN(dataTile_cor, gI);             
            cumsum_row_kernel<tile_shapeData_cor, tile_shapeSum_col><<<1, tile_shapeSum_col::ValidRow, 1>>>(SumTile_col.data(), OutTile_cor.data(), dataTile_cor.data(), oldSumTile_col.data());
            oldSumTile_col = SumTile_col;
            TCOPYOUT(gO, OutTile_cor);
        }        
    }
/*
    for(int i = 0; i < gIM; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
}

#endif
