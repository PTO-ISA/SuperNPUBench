#ifndef REDUCESUMTROWSUM_KERNEL_HPP
#define REDUCESUMTROWSUM_KERNEL_HPP

#ifndef __vbuf__
#define __vbuf__
#endif

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>

// ============================================================
// 文件说明：行方向求最大归约（Row Reduction - Max）基础实现
// ------------------------------------------------------------
// 作用：对 M×N 输入沿 N 轴（行方向）做求最大归约，输出 M×1。
// 分类：对应 README 中 "Row Reduction -> Basic Implementation"。
// 与 reducesum_rowvec.hpp 结构一致，仅把累加(+)替换为 blkv_max；
// 每个 lane 负责一行，沿列方向 8 路树形取 max，跨 tile 保持运行最大值。
// ============================================================

// ---- 单 tile 行最大归约 kernel：每个 lane 处理一行 ----
template<typename tileSrc, typename tileMax>
void __vec__ reducemax_row_kernel(
    typename tileMax::TileDType __out__ new_max,
    const typename tileSrc::TileDType __in__ src,
    const typename tileMax::TileDType __in__ old_max    
)
{
//    size_t i = blkv_get_index_x();  
    // j 为行号（lane 索引），每个 lane 独立取自己那一行的最大值
    size_t j = blkv_get_index_x();  
//    size_t j = blkv_get_index_y();
    size_t idx = j * tileMax::RowStride;    

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);   


    // 取上一 tile 的行最大作为初值，保证跨 tile 连续
    typename tileMax::DType upd_max = old_max_ptr[idx];


    // 8 路树形取 max：沿列方向每 8 列两两取 max 再合并
    #pragma clang loop unroll(full)
    for(size_t i=0;i<tileSrc::ValidCol;i+=8){
        size_t src_idx0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx1 =  (i+1) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx2 =  (i+2) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx3 =  (i+3) * tileSrc::ColStride + j * tileSrc::RowStride;        
        size_t src_idx4 =  (i+4) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx5 =  (i+5) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx6 =  (i+6) * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx7 =  (i+7) * tileSrc::ColStride + j * tileSrc::RowStride; 

        typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx0], src_ptr[src_idx1]);
        typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx2], src_ptr[src_idx3]);   
        typename tileMax::DType max_45 = blkv_max(src_ptr[src_idx4], src_ptr[src_idx5]);  
        typename tileMax::DType max_67 = blkv_max(src_ptr[src_idx6], src_ptr[src_idx7]);    

        typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
        typename tileMax::DType max_4567 = blkv_max(max_45, max_67);        

        typename tileMax::DType max_tmp = blkv_max(max_0123, max_4567);

        // 与历史最大值比较更新
        upd_max = blkv_max(upd_max, max_tmp);              
    }        


/*
    #pragma clang loop unroll(full)
    for(size_t i=0;i<tileSrc::ValidCol;i++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_max = blkv_max(upd_max, src_ptr[src_idx]);              
    }
*/    
    new_max_ptr[idx] = upd_max;    
}



// ------------------------------------------------------------
// 主机侧入口：行求最大归约（基础版）
// 流程与 reducesum_trowsum_rand 相同，仅 kernel 换为 max 版。
// ------------------------------------------------------------
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
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeMax = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>; // todo 尾块怎么处理？是否要作为参数写在这   
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeMax_col =  Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, rmd_M, 1>;      


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;                
    tile_shapeData_row dataTile_row;
    tile_shapeData_col dataTile_col;
    tile_shapeData_cor dataTile_cor;    
    
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;
    tile_shapeMax_col MaxTile_col;
    tile_shapeMax_col oldMaxTile_col;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

//    printf("tile_shapeSum::ValidCol = %d\n",  tile_shapeSum::ValidCol);
//    printf("tile_shapeSum::ValidRow = %d\n",  tile_shapeSum::ValidRow);    
//    printf("before for\n");
    // ---- 主区域：M 方向对齐的行块 ----
    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldMaxTile, 0);//初始化为0
        //初始化old_sum的tile      
        // 沿 N 方向逐 tile 拷入并取最大
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i);   
//            printf("before copy in , %d\n", i);                
            TCOPYIN(dataTile, gI);    
            reducemax_row_kernel<tile_shapeData, tile_shapeMax><<<tile_shapeMax::ValidRow, 1, 1>>>(MaxTile.data(), dataTile.data(), oldMaxTile.data());
//            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//            printf("kernel , %d\n", i);
            oldMaxTile = MaxTile;
        }
//        printf("end for%d\n",j);
        //for row corner
        // N 方向尾块
        if constexpr (rmd_N > 0){
            auto gI = gIIter(j, Nb);
            TCOPYIN(dataTile_row, gI);
            reducemax_row_kernel<tile_shapeData_row, tile_shapeMax><<<tile_shapeMax::ValidRow, 1, 1>>>(MaxTile.data(), dataTile_row.data(), oldMaxTile.data());            
//            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
            oldMaxTile = MaxTile;
        }
//        printf("before tcopyout\n");        
        TCOPYOUT(gO, MaxTile);
//        printf("end tcopyout\n"); 
    }
    //for col cor
    // ---- M 方向尾块（rmd_M>0）----
    if constexpr (rmd_M > 0){
        auto gO = gOIter(Mb, 0);
        TEXPANDSCALAR(oldMaxTile_col, 0);//初始化为0
        //初始化old_sum的tile      
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);
            TCOPYIN(dataTile_col, gI);
            reducemax_row_kernel<tile_shapeData_col, tile_shapeMax_col><<<tile_shapeMax_col::ValidRow, 1, 1>>>(MaxTile_col.data(), dataTile_col.data(), oldMaxTile_col.data());
            oldMaxTile_col = MaxTile_col;
        }
        if constexpr (rmd_N > 0){
            auto gI = gIIter(Mb, Nb);
            TCOPYIN(dataTile_cor, gI);             
            reducemax_row_kernel<tile_shapeData_cor, tile_shapeMax_col><<<tile_shapeMax_col::ValidRow, 1, 1>>>(MaxTile_col.data(), dataTile_cor.data(), oldMaxTile_col.data());
            oldMaxTile_col = MaxTile_col;
        }
        TCOPYOUT(gO, MaxTile_col);
    }
/*
    for(int i = 0; i < gIM; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
//    printf("end program\n"); 
}

#endif
