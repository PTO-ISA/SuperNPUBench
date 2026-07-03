#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ============================================================
// 文件说明：3D 列求和归约（Column Reduction - Sum）非对齐 120×8 专用版
// ------------------------------------------------------------
// 作用：针对形状如 120×8 这类 M 非二次幂的 3D 张量做列求和归约。
// 分类：对应 README 中 "3D Column Reduction Operators (Unalign Optimized)"。
// 特点：
//   1) 专为批量 3D 张量（Nums×gIM×gIN）设计，外层循环处理 Nums 个切片；
//   2) 对非二次幂行数（如 120 行）做布局变换：把 (gIM×gIN) 重排为
//      (gIM/8 × gIN*8)，以便凑齐 8 元向量树形累加；
//   3) 利用补零（TLOAD/TCOPYIN 默认补 0）让非 valid 区也参与 8 路归约，
//      不影响结果且无需特殊尾块处理。
// 两阶段结构：
//   reducesum_col_tmp()   : 每 tile 列归约到临时结果；
//   reducesum_col_final() : 把临时结果与累加器 old_sum 合并产出最终值。
// 模板参数：tM_VLD 为 tile 的有效行数（非对齐场景关键参数）。
// ============================================================


// ==============================================
// ==============================================
// ---- 阶段1：单 tile 内 8 路树形列归约，输出临时和 ----
template<typename tileSrc, typename tileTmp>
void __vec__ reducesum_col_tmp(
    typename tileTmp::TileDType __out__ tmp_sum,
    const typename tileSrc::TileDType __in__ src    
)
{
    // 当前 lane 负责一列
    size_t i = blkv_get_index_x();  

    __vbuf__ typename tileTmp::DType *tmp_sum_ptr = blkv_get_tile_ptr(tmp_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
//    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);   

    // 局部累加器清零
    typename tileTmp::DType upd_tmp_sum = 0;
   
    // 注意：此处用 tileSrc::Rows（而非 ValidRow），让非 valid 的补零区也参与计算，
    // 从而凑出完整的 8 元树形累加，省去尾块特判
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
        // 8 路两两求和 -> 树形合并
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

// ---- 阶段2：把临时和与历史累加器 old_sum 合并，写出最终列和 ----
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


    // 取历史累加值（跨切片/跨 tile 的累积）
    typename tileSum::DType upd_sum = old_sum_ptr[i];
   

    // 对临时 tile 中 8 个切片结果再做一次 8 路树形合并
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
    // 合并临时结果与历史累加值
    new_sum_ptr[i] = sum_all + upd_sum;   
}




// ------------------------------------------------------------
// 主机侧入口：3D 非对齐列求和归约（120×8 专用）
// 布局变换：输入按 (gIM/8, gIN*8) 视图重排，便于 8 路向量化。
// 调用方需对 Nums 个 3D 切片循环调用本函数。
// ------------------------------------------------------------
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

    // 关键布局变换：(gIM×gIN) -> (gIM/8 × gIN*8)，使每 8 行拍平到列上做向量化
    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM/8, gIN*8>>;     // 
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;    
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    // tile 行维度也除以 8、列维度乘 8；tM_VLD/8 为有效行（用于非对齐补零）
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


    // 单 tile 流程：拷入（自动补零）-> 阶段1 tmp 归约 -> 阶段2 final 合并 -> 拷出
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
