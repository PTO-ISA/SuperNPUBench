#ifndef REDUCESUMCOLVEC_KERNEL_HPP
#define REDUCESUMCOLVEC_KERNEL_HPP


#include <common/pto_tileop.hpp>
#include "template_asm.h"


using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


// ============================================================
// 文件说明：3D 列求最大归约（Column Reduction - Max）非对齐 120×8 专用版
// ------------------------------------------------------------
// 作用：对形状如 120×8 的 3D 张量做列求最大归约（沿 M 轴取每列最大值）。
// 分类：对应 README 中 "3D Column Reduction Operators (Unalign Optimized)"。
// 与 reducesum_colvec_unalign_120_8.hpp 结构一致，仅把求和(+)替换为 blkv_max。
// 关键点：
//   1) 布局变换 (gIM×gIN) -> (gIM/8 × gIN*8)，凑齐 8 元向量；
//   2) 用补零让非 valid 区参与 8 路树形取 max（0 不影响 max 结果前提是数据非负，
//      实际依赖 TLOAD 默认补 0）；
//   3) 两阶段：reducemax_col_tmp() 每 tile 取临时最大；reducemax_col_final() 合并。
// ============================================================


// ==============================================
// ==============================================
// ---- 阶段1：单 tile 内 8 路树形列取最大，输出临时最大值 ----
template<typename tileSrc, typename tileTmp>
void __vec__ reducemax_col_tmp(
    typename tileTmp::TileDType __out__ tmp_max,
    const typename tileSrc::TileDType __in__ src
)
{
    // 当前 lane 负责一列
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileTmp::DType *tmp_max_ptr = blkv_get_tile_ptr(tmp_max);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
//    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);

    // 局部最大值初值
    typename tileTmp::DType upd_tmp_max = 0;

    // 用 Rows（非 valid 补零区也参与）凑齐 8 元树形取 max
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
        // 8 路两两取 max -> 树形合并
        typename tileTmp::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
        typename tileTmp::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
        typename tileTmp::DType max_45 = blkv_max(src_ptr[src_idx_4], src_ptr[src_idx_5]);
        typename tileTmp::DType max_67 = blkv_max(src_ptr[src_idx_6], src_ptr[src_idx_7]);
        typename tileTmp::DType max_0123 = blkv_max(max_01, max_23);
        typename tileTmp::DType max_4567 = blkv_max(max_45, max_67);
        typename tileTmp::DType max_tmp = blkv_max(max_0123, max_4567);
        upd_tmp_max = blkv_max(upd_tmp_max, max_tmp);
    }

    tmp_max_ptr[i] = upd_tmp_max;
}

// ---- 阶段2：把临时最大值与历史累加器 old_max 合并，写出最终列最大 ----
template<typename tileTmp, typename tileMax>
void __vec__ reducemax_col_final(
    typename tileMax::TileDType __out__ new_max,
    const typename tileTmp::TileDType __in__ src,
    const typename tileMax::TileDType __in__ old_max
)
{
    size_t i = blkv_get_index_x();

    __vbuf__ typename tileMax::DType *new_max_ptr = blkv_get_tile_ptr(new_max);
    __vbuf__ typename tileTmp::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileMax::DType *old_max_ptr = blkv_get_tile_ptr(old_max);


    // 取历史最大值
    typename tileMax::DType upd_max = old_max_ptr[i];


    // 对临时 tile 中 8 个切片结果做 8 路树形取 max
    size_t src_idx_0 =  i * tileMax::ColStride + 0 * tileMax::ValidCol;
    size_t src_idx_1 =  i * tileMax::ColStride + 1 * tileMax::ValidCol;
    size_t src_idx_2 =  i * tileMax::ColStride + 2 * tileMax::ValidCol;
    size_t src_idx_3 =  i * tileMax::ColStride + 3 * tileMax::ValidCol;
    size_t src_idx_4 =  i * tileMax::ColStride + 4 * tileMax::ValidCol;
    size_t src_idx_5 =  i * tileMax::ColStride + 5 * tileMax::ValidCol;
    size_t src_idx_6 =  i * tileMax::ColStride + 6 * tileMax::ValidCol;
    size_t src_idx_7 =  i * tileMax::ColStride + 7 * tileMax::ValidCol;
    typename tileMax::DType max_01 = blkv_max(src_ptr[src_idx_0], src_ptr[src_idx_1]);
    typename tileMax::DType max_23 = blkv_max(src_ptr[src_idx_2], src_ptr[src_idx_3]);
    typename tileMax::DType max_45 = blkv_max(src_ptr[src_idx_4], src_ptr[src_idx_5]);
    typename tileMax::DType max_67 = blkv_max(src_ptr[src_idx_6], src_ptr[src_idx_7]);
    typename tileMax::DType max_0123 = blkv_max(max_01, max_23);
    typename tileMax::DType max_4567 = blkv_max(max_45, max_67);
    typename tileMax::DType max_all = blkv_max(max_0123, max_4567);

//        upd_max = upd_max + max_tmp;


    // 合并临时结果与历史最大值
    new_max_ptr[i] = blkv_max(max_all, upd_max);
}




// ------------------------------------------------------------
// 主机侧入口：3D 非对齐列求最大归约（120×8 专用）
// 布局变换与 reducesum 版一致。
// ------------------------------------------------------------
template<typename dtype, int gIM, int gIN, int tM, int tN, int tM_VLD>
void reducemax_col_rand(
    dtype *in_ptr,
//    dtype *inzero_ptr,
    dtype *out_ptr
)
{

//    const int Mb = (gIM/8) / tM;

    const int rmd_M = gIM % tM;
    const int rmd_N = gIN % tN;
//    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    // 布局变换：(gIM×gIN) -> (gIM/8 × gIN*8)
    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM/8, gIN*8>>;     //
//    using gm_shapeMax = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;
    using tile_shapeData = Tile<Location::Vec, dtype, tM/8, tN*8, BLayout::RowMajor, tM_VLD/8, tN*8>; //
//    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor,rmd_M, tN>; //
    using tile_shapeTmp = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor>; //
    using tile_shapeMax = Tile<Location::Vec, dtype, 1, tN*8, BLayout::RowMajor, 1, tN>; //



//    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; //
//    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; //
//    using tile_shapeMax_row = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor, 1, rmd_N>; //
    //need tM = 1;


    gm_shapeIn inGm(in_ptr);
//    gm_shapeOut ZeroGm(inzero_ptr);
    gm_shapeOut outGm(out_ptr);
//    gm_shapeMax olcMaxGm(old_max_ptr);

    tile_shapeData dataTile;
//    tile_shapeData_col dataTile_col;
    tile_shapeTmp TmpTile;
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;

//    tile_shapeData_row dataTile_row;
//    tile_shapeData_cor dataTile_cor;
//    tile_shapeMax_row MaxTile_row;
//    tile_shapeMax_row oldMaxTile_row;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itIn_row = global_iterator<gm_shapeIn, tile_shapeMax>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn  gIIter(in_ptr);
    itIn_row  gIIter_rmd_row(in_ptr);
//    itZero  gZeroIter(inzero_ptr);
    itOut gOIter(out_ptr);


    // 单 tile 流程：拷入(补零) -> 阶段1 tmp 取最大 -> 阶段2 final 合并 -> 拷出
    auto gO = gOIter(0, 0);
    TEXPANDSCALAR(oldMaxTile, 0);//初始化为0
    auto gI = gIIter(0, 0);
    TLOAD(dataTile, gI);//补0的TLOAD
    reducemax_col_tmp<tile_shapeData, tile_shapeTmp><<<tile_shapeTmp::ValidCol, tile_shapeTmp::ValidRow, 1>>>(TmpTile.data(), dataTile.data());
    reducemax_col_final<tile_shapeTmp, tile_shapeMax><<<tile_shapeMax::ValidCol, tile_shapeMax::ValidRow, 1>>>(MaxTile.data(), TmpTile.data(), oldMaxTile.data());
    oldMaxTile = MaxTile;
    TSTORE(gO, MaxTile);
}

#endif
