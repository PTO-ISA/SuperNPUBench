#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ============================================================
// 文件说明：列方向求和归约（Column Reduction - Sum）基础实现
// ------------------------------------------------------------
// 作用：对 M×N 的输入矩阵沿 M 轴（列方向/垂直方向）做求和归约，
//       输出 1×N 的行向量。
// 分类：对应 README 中的 "Column Reduction Operators -> Basic Implementation"。
// 关键特性：
//   1) 采用 8 路展开的树形累加（8-way tree reduction），提升指令级并行；
//   2) 同时处理对齐主块与尾块（remainder/corner tile）的非对齐维度；
//   3) 使用 Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor> 进行分块管理。
// 模板参数：
//   dtype : 数据类型（int32_t / float 等）
//   gIM   : 全局输入行数（M 维）
//   gIN   : 全局输入列数（N 维）
//   tM    : 分块的行尺寸
//   tN    : 分块的列尺寸
// ============================================================
template<typename tileSrc, typename tileSum>
void __vec__ reducesum_col_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileSum::TileDType __in__ old_sum
)
{
    // 取当前向量单元在 x 方向的 lane 索引：每个 lane 负责一列的累加
    size_t i = blkv_get_index_x();

    // 获取各 tile 在向量内存中的基地址指针
    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);


//    typename tileSum::DType upd_sum = old_sum_ptr[i];
    // 本 tile 内的局部累加器，先清零（与历史 old_sum 在最后再合并）
    typename tileSum::DType upd_sum = 0;

    // 8 路展开的树形归约：每次同时取 8 个相邻行做两两求和，再树形合并
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j+=8){
        // 计算同一列上连续 8 行的源数据偏移（ColStride 为列步长，RowStride 为行步长）
        size_t src_idx_0 =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        size_t src_idx_1 =  i * tileSrc::ColStride + (j + 1) * tileSrc::RowStride;
        size_t src_idx_2 =  i * tileSrc::ColStride + (j + 2) * tileSrc::RowStride;
        size_t src_idx_3 =  i * tileSrc::ColStride + (j + 3) * tileSrc::RowStride;
        size_t src_idx_4 =  i * tileSrc::ColStride + (j + 4) * tileSrc::RowStride;
        size_t src_idx_5 =  i * tileSrc::ColStride + (j + 5) * tileSrc::RowStride;
        size_t src_idx_6 =  i * tileSrc::ColStride + (j + 6) * tileSrc::RowStride;
        size_t src_idx_7 =  i * tileSrc::ColStride + (j + 7) * tileSrc::RowStride;
        // 第一层：4 组两两求和
        typename tileSum::DType sum_01 = src_ptr[src_idx_0] + src_ptr[src_idx_1];
        typename tileSum::DType sum_23 = src_ptr[src_idx_2] + src_ptr[src_idx_3];
        typename tileSum::DType sum_45 = src_ptr[src_idx_4] + src_ptr[src_idx_5];
        typename tileSum::DType sum_67 = src_ptr[src_idx_6] + src_ptr[src_idx_7];
        // 第二层：2 组两两求和
        typename tileSum::DType sum_0123 = sum_01 + sum_23;
        typename tileSum::DType sum_4567 = sum_45 + sum_67;
        // 第三层：最终 8 元合并
        typename tileSum::DType sum_tmp = sum_0123 + sum_4567;
        // 累加到本 tile 的局部和上
        upd_sum = upd_sum + sum_tmp;
    }

/*
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_sum = upd_sum + src_ptr[src_idx];
    }
*/
    // 与上一 tile 的 old_sum 合并，写出本列的新累加结果
    new_sum_ptr[i] = upd_sum + old_sum_ptr[i];
}




// ------------------------------------------------------------
// 主机侧入口：列求和归约（基础版）
// 输入：in_ptr 指向 gIM×gIN 的全局矩阵；输出：out_ptr 指向 1×gIN 的结果
// 流程：按 N 方向逐列块处理，每列块内沿 M 方向逐 tile 累加；
//       同时分别处理 M、N 两个方向的尾块（非对齐部分）。
// ------------------------------------------------------------
template<typename dtype, int gIM, int gIN, int tM, int tN>
void reducesum_colsum_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,
    dtype *out_ptr
)
{

    // 主块数量与尾块余数（用于非对齐维度的边角处理）
    const int Mb = gIM / tM;
    const int Nb = gIN / tN;

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    // 全局输入/输出张量形状定义
    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    // 各类分块形状：主块、列尾块（M 不对齐）、行尾块（N 不对齐）、角块（M、N 都不对齐）
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //
    using tile_shapeSum = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; //



    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; //
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //
    using tile_shapeSum_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; //
    //need tM = 1;


    // 实例化全局张量与本地 tile
    gm_shapeIn inGm(in_ptr);
//    gm_shapeOut ZeroGm(inzero_ptr);
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeSum_row SumTile_row;
    tile_shapeSum_row oldSumTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    // 全局迭代器：按 tile 粒度遍历全局内存
    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeSum>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);
    itOut gOIter(out_ptr);

//    dtype zero = 0;

    // ---- 主区域：N 方向对齐的列块 ----
    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
        auto gO = gOIter(0, j);
        TEXPANDSCALAR(oldSumTile, 0);//初始化为0
//        TLOAD(oldSumTile, gZero);//初始化为0
        //初始化old_sum的tile
        //need
        // 沿 M 方向逐 tile 拷入并累加
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            reducesum_col_kernel<tile_shapeData, tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
            oldSumTile = SumTile;
        }
        // 处理 M 方向尾块（rmd_M>0 时使用列尾块形状）
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_col, gI);
            reducesum_col_kernel<tile_shapeData_col,tile_shapeSum><<<tile_shapeSum::ValidCol, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile_col.data(), oldSumTile.data());
            oldSumTile = SumTile;
        }
        TSTORE(gO, SumTile);
    }
    // ---- N 方向尾块区域（rmd_N>0）：使用行尾块/角块形状 ----
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldSumTile_row, 0);//初始化为0
//        TLOAD(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            reducesum_col_kernel<tile_shapeData_row,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_row.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reducesum_col_kernel<tile_shapeData_cor,tile_shapeSum_row><<<tile_shapeSum_row::ValidCol, tile_shapeSum_row::ValidRow, 1>>>(SumTile_row.data(), dataTile_cor.data(), oldSumTile_row.data());
            oldSumTile_row = SumTile_row;
        }
        TSTORE(gO, SumTile_row);
    }
}

#endif
