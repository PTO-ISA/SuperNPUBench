#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ============================================================
// 文件说明：列方向求积归约（Column Reduction - Prod）基础实现
// ------------------------------------------------------------
// 作用：对 M×N 输入沿 M 轴（列方向）做求积归约，输出 1×N。
// 分类：对应 README 中 "Column Reduction -> Basic Implementation"（prod 变体）。
// 与 reducesum_colvec.hpp 结构一致，仅把累加(+)替换为逐元素乘法(*)。
// 注意：求积不适用 8 路树形展开（乘法不满足可加的树形加速收益结构），
//       因此本 kernel 采用标量循环逐元素连乘。
// ============================================================


// ==============================================
// ==============================================
// ---- 单 tile 列求积归约 kernel：每个 lane 处理一列 ----
template<typename tileSrc, typename timeProd>
void __vec__ reduceprod_col_kernel(
    typename timeProd::TileDType __out__ new_prod,
    const typename tileSrc::TileDType __in__ src,
    const typename timeProd::TileDType __in__ old_prod
)
{
    // 当前 lane 索引：每个 lane 负责一列的连乘
    size_t i = blkv_get_index_x();

    __vbuf__ typename timeProd::DType *new_prod_ptr = blkv_get_tile_ptr(new_prod);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename timeProd::DType *old_prod_ptr = blkv_get_tile_ptr(old_prod);


    // 取上一 tile 的列积作为初值，保证跨 tile 连乘连续
    typename timeProd::DType upd_prod = old_prod_ptr[i];

    // 逐元素连乘：沿列方向把每个元素乘到累乘器上
    #pragma clang loop unroll(full)
    for(size_t j=0;j<tileSrc::ValidRow;j++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_prod = upd_prod * src_ptr[src_idx];
    }
    new_prod_ptr[i] = upd_prod;
}



// ------------------------------------------------------------
// 主机侧入口：列求积归约（基础版）
// 流程与 reducesum_colsum_rand 相同，仅 kernel 换为 prod 版。
// ------------------------------------------------------------
template<typename dtype, int gIM, int gIN, int tM, int tN>
void reduceprod_col_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,
    dtype *out_ptr
)
{

    const int Mb = gIM / tM;
    const int Nb = gIN / tN;

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM, gIN>>;     //
//    using gm_shapeSum = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>; //
    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>; //
    using tile_shapeProd = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>; //



    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; //
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //
    using tile_shapeProd_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; //
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);
//    gm_shapeOut ZeroGm(inzero_ptr);
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);

    tile_shapeData dataTile;
    tile_shapeData_col dataTile_col;
    tile_shapeProd ProdTile;
    tile_shapeProd oldProdTile;

    tile_shapeData_row dataTile_row;
    tile_shapeData_cor dataTile_cor;
    tile_shapeProd_row ProdTile_row;
    tile_shapeProd_row oldProdTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeProd>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeProd>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);
    itOut gOIter(out_ptr);

//    dtype zero = 0;

    // ---- 主区域：N 方向对齐的列块 ----
    for (int j = 0; j < Nb; ++j) {
//        auto gZero = gZeroIter(0, j);
        auto gO = gOIter(0, j);
        TEXPANDSCALAR(oldProdTile, 0);//初始化为0
//        TLOAD(oldSumTile, gZero);//初始化为0
        //初始化old_sum的tile
        //need
        // 沿 M 方向逐 tile 拷入并连乘
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, j);
            TLOAD(dataTile, gI);
            reduceprod_col_kernel<tile_shapeData, tile_shapeProd><<<tile_shapeProd::ValidCol, tile_shapeProd::ValidRow, 1>>>(ProdTile.data(), dataTile.data(), oldProdTile.data());
            oldProdTile = ProdTile;
        }
        // M 方向尾块
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, j);
            TLOAD(dataTile_col, gI);
            reduceprod_col_kernel<tile_shapeData_col,tile_shapeProd><<<tile_shapeProd::ValidCol, tile_shapeProd::ValidRow, 1>>>(ProdTile.data(), dataTile_col.data(), oldProdTile.data());
            oldProdTile = ProdTile;
        }
        TSTORE(gO, ProdTile);
    }
    // ---- N 方向尾块区域 ----
    if constexpr (rmd_N > 0){
//        auto gZero = gZeroIter(0, Nb);
        auto gO = gOIter(0, Nb);
        TEXPANDSCALAR(oldProdTile_row, 0);//初始化为0
//        TLOAD(oldSumTile_row, gZero);//初始化为0
        for (int i = 0; i < Mb; ++i) {
            auto gI = gIIter(i, Nb);
            TLOAD(dataTile_row, gI);
            reduceprod_col_kernel<tile_shapeData_row,tile_shapeProd_row><<<tile_shapeProd_row::ValidCol, tile_shapeProd_row::ValidRow, 1>>>(ProdTile_row.data(), dataTile_row.data(), oldProdTile_row.data());
            oldProdTile_row = ProdTile_row;
        }
        if constexpr (rmd_M > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reduceprod_col_kernel<tile_shapeData_cor,tile_shapeProd_row><<<tile_shapeProd_row::ValidCol, tile_shapeProd_row::ValidRow, 1>>>(ProdTile_row.data(), dataTile_cor.data(), oldProdTile_row.data());
            oldProdTile_row = ProdTile_row;
        }
        TSTORE(gO, ProdTile_row);
    }
}

#endif
