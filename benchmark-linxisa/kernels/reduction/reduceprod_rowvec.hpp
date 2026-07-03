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
// 文件说明：行方向求积归约（Row Reduction - Prod）基础实现
// ------------------------------------------------------------
// 作用：对 M×N 输入沿 N 轴（行方向）做求积归约，输出 M×1。
// 分类：对应 README 中 "Row Reduction -> Basic Implementation"（prod 变体）。
// 与 reducesum_rowvec.hpp 结构一致，仅把累加(+)替换为逐元素乘法(*)。
// 每个 lane 负责一行连乘，跨 tile 保持运行积。
// 注意：求积采用标量循环逐元素连乘（不使用 8 路树形）。
// ============================================================

// ---- 单 tile 行求积归约 kernel：每个 lane 处理一行 ----
template<typename tileSrc, typename tileProd>
void __vec__ reduceprod_row_kernel(
    typename tileProd::TileDType __out__ new_prod,
    const typename tileSrc::TileDType __in__ src,
    const typename tileProd::TileDType __in__ old_prod    
)
{
//    size_t i = blkv_get_index_x();  
    // j 为行号（lane 索引），每个 lane 独立连乘自己那一行
    size_t j = blkv_get_index_x();  
//    size_t j = blkv_get_index_y();
    size_t idx = j * tileProd::RowStride;    

    __vbuf__ typename tileProd::DType *new_prod_ptr = blkv_get_tile_ptr(new_prod);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileProd::DType *old_prod_ptr = blkv_get_tile_ptr(old_prod);   


    // 取上一 tile 的行积作为初值，保证跨 tile 连乘连续
    typename tileProd::DType upd_prod = old_prod_ptr[idx];

    // 逐元素连乘：沿列方向把每个元素乘到累乘器上
    #pragma clang loop unroll(full)
    for(size_t i=0;i<tileSrc::ValidCol;i++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_prod = upd_prod * src_ptr[src_idx];              
    }    
    new_prod_ptr[idx] = upd_prod;    
}



// ------------------------------------------------------------
// 主机侧入口：行求积归约（基础版）
// 流程与 reducesum_trowsum_rand 相同，仅 kernel 换为 prod 版。
// ------------------------------------------------------------
template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void reduceprod_row_rand(
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
    using tile_shapeData_row = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeProd = Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, tM, 1>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec


    using tile_shapeData_col = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>; // todo 尾块怎么处理？是否要作为参数写在这   
    using tile_shapeData_cor = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeProd_col =  Tile<Location::Vec, dtype, tM, 8, BLayout::RowMajor, rmd_M, 1>;      


    gm_shapeIn inGm(in_ptr);    
    gm_shapeOut outGm(out_ptr);
//    gm_shapeSum olcSumGm(old_sum_ptr);    

    tile_shapeData dataTile;                
    tile_shapeData_row dataTile_row;
    tile_shapeData_col dataTile_col;
    tile_shapeData_cor dataTile_cor;    
    
    tile_shapeProd ProdTile;
    tile_shapeProd oldProdTile;
    tile_shapeProd_col ProdTile_col;
    tile_shapeProd_col oldProdTile_col;    

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;  
    using itOut = global_iterator<gm_shapeOut, tile_shapeProd>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

//    printf("tile_shapeSum::ValidCol = %d\n",  tile_shapeSum::ValidCol);
//    printf("tile_shapeSum::ValidRow = %d\n",  tile_shapeSum::ValidRow);    
//    printf("before for\n");
    // ---- 主区域：M 方向对齐的行块 ----
    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldProdTile, 0);//初始化为0
        //初始化old_sum的tile      
        // 沿 N 方向逐 tile 拷入并连乘
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i);   
//            printf("before copy in , %d\n", i);                
            TCOPYIN(dataTile, gI);    
            reduceprod_row_kernel<tile_shapeData, tile_shapeProd><<<tile_shapeProd::ValidRow, 1, 1>>>(ProdTile.data(), dataTile.data(), oldProdTile.data());
//            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//            printf("kernel , %d\n", i);
            oldProdTile = ProdTile;
        }
//        printf("end for%d\n",j);
        //for row corner
        // N 方向尾块
        if constexpr (rmd_N > 0){
            auto gI = gIIter(j, Nb);
            TCOPYIN(dataTile_row, gI);
            reduceprod_row_kernel<tile_shapeData_row, tile_shapeProd><<<tile_shapeProd::ValidRow, 1, 1>>>(ProdTile.data(), dataTile_row.data(), oldProdTile.data());            
//            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
            oldProdTile = ProdTile;
        }
//        printf("before tcopyout\n");        
        TCOPYOUT(gO, ProdTile);
//        printf("end tcopyout\n"); 
    }
    //for col cor
    // ---- M 方向尾块（rmd_M>0）----
    if constexpr (rmd_M > 0){
        auto gO = gOIter(Mb, 0);
        TEXPANDSCALAR(oldProdTile_col, 0);//初始化为0
        //初始化old_sum的tile      
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);   
            TCOPYIN(dataTile_col, gI);                  
            reduceprod_row_kernel<tile_shapeData_col, tile_shapeProd_col><<<tile_shapeProd_col::ValidRow, 1, 1>>>(ProdTile_col.data(), dataTile_col.data(), oldProdTile_col.data());
            oldProdTile_col = ProdTile_col;
        }
        if constexpr (rmd_N > 0){
            auto gI = gIIter(Mb, Nb);
            TCOPYIN(dataTile_cor, gI);             
            reduceprod_row_kernel<tile_shapeData_cor, tile_shapeProd_col><<<tile_shapeProd_col::ValidRow, 1, 1>>>(ProdTile_col.data(), dataTile_cor.data(), oldProdTile_col.data());
            oldProdTile_col = ProdTile_col;
        }
        TCOPYOUT(gO, ProdTile_col);
    }
/*
    for(int i = 0; i < gIM; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
//    printf("end program\n"); 
}

#endif
