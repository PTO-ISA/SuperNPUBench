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

// ============================================================
// 文件说明：行方向求和归约（Row Reduction - Sum）优化版 single_tree
// ------------------------------------------------------------
// 作用：对 M×N 输入沿 N 轴做行求和归约，输出 M×1。
// 分类：对应 README 中 "Row Reduction -> Optimized Single Tree"。
// 优化策略（两阶段）：
//   1) 中间结果采用列主序(ColMajor)存储，提升访存效率、降低带宽消耗；
//   2) 用 4 路并行处理（blkv_get_index_y() 提供 z 维索引），把 tile 的列
//      分成 4 段并行归约；
//   3) 多级树形归约：8 路 -> 64 路(8×8) -> 步长倍增渐进式合并；
//   4) reducesum_row_kernel() 写中间 tile，reducesum_row_final_kernel() 做最终合并。
// ============================================================

// ---- 阶段1：单 tile 行归约（4 路并行 + 多级树形）----
// 参数：new_sum 输出中间 tile；src 输入；src_col 列主序中间缓冲；old_sum 历史；tile_idx 槽位号
template<typename tileSrc, typename tileSrcCol, typename tileTmpSum>
void __vec__ reducesum_row_kernel(
    typename tileTmpSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileSrcCol::TileDType __in__ src_col,
    const typename tileTmpSum::TileDType __in__ old_sum,
    const size_t tile_idx    
)
{

    // j 为行 lane，z 为 4 路并行的段索引
    size_t j = blkv_get_index_x();  
    size_t z = blkv_get_index_y();     
    // 本段在 src / src_col 中的基地址偏移（按 ValidCol/4 划分 4 段）
    size_t stride_src = z * (tileSrc::ValidCol/4) * tileSrc::ColStride;
    size_t stride_src_col = z * (tileSrcCol::ValidCol/4) * tileSrcCol::ColStride;    
  
    __vbuf__ typename tileTmpSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSrc::DType *src_col_ptr = blkv_get_tile_ptr(src_col);    
    __vbuf__ typename tileTmpSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   

/*
    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpSum::ValidCol/4;i++){
        size_t old_sum_idx =  z * tileTmpSum::ValidCol/4 * tileTmpSum::ColStride + i * tileTmpSum::ColStride + j * tileTmpSum::RowStride;       
        new_sum_ptr[old_sum_idx] = old_sum_ptr[old_sum_idx];          
    }    
*/

    // 第一级：8 路树形归约，每 8 列合并为 1 个结果写入 src_col 缓冲
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

        typename tileSrc::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileSrc::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];   
        typename tileSrc::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];  
        typename tileSrc::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];    

        typename tileSrc::DType sum_0123 = sum_01 + sum_23;
        typename tileSrc::DType sum_4567 = sum_45 + sum_67;        

        typename tileSrc::DType sum_all = sum_0123 + sum_4567;

        // 写入列主序缓冲的对应位置
        size_t src_col_idx_0 = stride_src_col + (i/8) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
        src_col_ptr[src_col_idx_0] = sum_all;         
    }        


    // 第二级：对 src_col 缓冲再做 8 路归约（64 路 = 8×8）
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
        typename tileSrcCol::DType tmp_sum_01 = src_col_ptr[tmp_idx_0]+ src_col_ptr[tmp_idx_1];
        typename tileSrcCol::DType tmp_sum_23 = src_col_ptr[tmp_idx_2]+ src_col_ptr[tmp_idx_3]; 
        typename tileSrcCol::DType tmp_sum_45 = src_col_ptr[tmp_idx_4]+ src_col_ptr[tmp_idx_5]; 
        typename tileSrcCol::DType tmp_sum_67 = src_col_ptr[tmp_idx_6]+ src_col_ptr[tmp_idx_7];  
        typename tileSrcCol::DType tmp_sum_0123 = tmp_sum_01 + tmp_sum_23; 
        typename tileSrcCol::DType tmp_sum_4567 = tmp_sum_45 + tmp_sum_67; 
        typename tileSrcCol::DType tmp_sum_all = tmp_sum_0123 + tmp_sum_4567;
        src_col_ptr[tmp_idx_0] = tmp_sum_all;
    }



    // 第三级：步长倍增渐进式归约（iternum = ctz(ValidCol/4) - 3）
    size_t stride = 8;
    size_t iternum = __builtin_ctz(tileSrcCol::ValidCol/4) - 3;

    #pragma clang loop unroll(full) 
    for(size_t k=0; k<iternum; k++){
        //#pragma clang loop unroll(full) 
        for(size_t i=0; i<tileSrcCol::ValidCol/4; i+=(stride*2)){
            size_t src_idx_0 =  stride_src_col + (i + 0*stride) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
            size_t src_idx_1 =  stride_src_col + (i + 1*stride) * tileSrcCol::ColStride + j * tileSrcCol::RowStride;
            typename  tileSrcCol::DType sum_01 = src_col_ptr[src_idx_0] + src_col_ptr[src_idx_1];           
            src_col_ptr[src_idx_0] = sum_01;          
        }
        stride = stride*2;
    }


    // 把 old_sum 原值拷到 new_sum（保留历史值）
    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpSum::ValidCol/4;i++){
        size_t old_sum_idx =  z * tileTmpSum::ValidCol/4 * tileTmpSum::ColStride + i * tileTmpSum::ColStride + j * tileTmpSum::RowStride;       
        new_sum_ptr[old_sum_idx] = old_sum_ptr[old_sum_idx];          
    }    


    // 将本段最终归约结果写入中间 tile 的对应槽位 tile_idx
    size_t src_sum_idx = stride_src_col + j * tileSrcCol::RowStride;
    size_t sum_tile_idx = z * tileTmpSum::ValidCol/4 * tileTmpSum::ColStride + tile_idx * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
    new_sum_ptr[sum_tile_idx] = src_col_ptr[src_sum_idx];  
}


// ---- 阶段2：对中间 tile 做最终行归约，产出 M×1 ----
template<typename tileTmpSum, typename tileSum>
void __vec__ reducesum_row_final_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileTmpSum::TileDType __in__ tmp_sum
){
    size_t j = blkv_get_index_x();
    size_t idx = j * tileSum::RowStride;

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileTmpSum::DType *tmp_sum_ptr = blkv_get_tile_ptr(tmp_sum);

    // 第一级：8 路归约
    #pragma clang loop unroll(full) 
    for(size_t i=0;i<tileTmpSum::Cols;i+=8){
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

    // 第二级：步长倍增渐进式归约
    size_t stride = 8;
    size_t iternum = __builtin_ctz(tileTmpSum::Cols) - 3;    
    #pragma clang loop unroll(full) 
    for(size_t k=0;k<iternum;k++){
        #pragma clang loop unroll(full) 
        for(size_t i=0;i<tileTmpSum::Cols;i+=(stride*2)){
            size_t src_idx_0 =  (i + 0*stride) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
            size_t src_idx_1 =  (i + 1*stride) * tileTmpSum::ColStride + j * tileTmpSum::RowStride;
            typename  tileTmpSum::DType sum_01 = tmp_sum_ptr[src_idx_0] + tmp_sum_ptr[src_idx_1];           
            tmp_sum_ptr[src_idx_0] = sum_01;          
        }
        stride = stride*2;
    }    

    // 写出每行最终和
    size_t sum_idx = j * tileTmpSum::RowStride;
    new_sum_ptr[idx] = tmp_sum_ptr[sum_idx];
}


// ------------------------------------------------------------
// 主机侧入口：行求和归约（single_tree 优化版）
// 注意：此版本只处理对齐主区域，未含 rmd 尾块分支。
// ------------------------------------------------------------
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
    // 列主序中间缓冲：tN/8 列，便于 8 路归约后存储
    using tile_shapeDataCol = Tile<Location::Vec, dtype, tM, tN/8, BLayout::ColMajor>; // todo 尾块怎么处理？是否要作为参数写在这    
    using tile_shapeSum = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec
    // 中间 tile：列主序，64 列、有效列=Nb*4（4 路并行 × Nb 个 tile）
    using tile_shapeTmpSum = Tile<Location::Vec, dtype, tM, 64, BLayout::ColMajor, tM, Nb*4>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;   
    tile_shapeDataCol dataTile_col;                        
    tile_shapeSum SumTile;
    tile_shapeTmpSum oldtmpSumTile;
    tile_shapeTmpSum tmpSumTile;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);


    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldtmpSumTile, 0);//初始化为0  
    TEXPANDSCALAR(dataTile_col, 0);//初始化为0     
    // 阶段1：逐 tile 拷入并做 4 路并行多级树形归约，写入中间 tile
    for (int i = 0; i < Nb; ++i) {
        auto gI = gIIter(0, i);                
        TCOPYIN(dataTile, gI);    
        reducesum_row_kernel<tile_shapeData, tile_shapeDataCol, tile_shapeTmpSum><<<tile_shapeTmpSum::ValidRow, 4, 1>>>(tmpSumTile.data(), 
                                                                                                                        dataTile.data(), 
                                                                                                                        dataTile_col.data(), 
                                                                                                                        oldtmpSumTile.data(),
                                                                                                                        i);
        oldtmpSumTile = tmpSumTile;
    }
    // 阶段2：最终归约并写出
    reducesum_row_final_kernel<tile_shapeTmpSum, tile_shapeSum><<<tile_shapeTmpSum::ValidRow, 1, 1>>>(SumTile.data(), 
                                                                                                      tmpSumTile.data());     
    TCOPYOUT(gO, SumTile);
}

#endif
