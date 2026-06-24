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

template<typename tileSrc, typename tileProd>
void __vec__ reduceprod_row_kernel(
    typename tileProd::TileDType __out__ new_prod,
    const typename tileSrc::TileDType __in__ src,
    const typename tileProd::TileDType __in__ old_prod
)
{
//    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_x();
//    size_t j = blkv_get_index_y();
    size_t idx = j * tileProd::RowStride;

    __vbuf__ typename tileProd::DType *new_prod_ptr = blkv_get_tile_ptr(new_prod);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileProd::DType *old_prod_ptr = blkv_get_tile_ptr(old_prod);


    typename tileProd::DType upd_prod = old_prod_ptr[idx];

    #pragma clang loop unroll(full)
    for(size_t i=0;i<tileSrc::ValidCol;i++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_prod = upd_prod * src_ptr[src_idx];
    }
    new_prod_ptr[idx] = upd_prod;
}



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
    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldProdTile, 0);//初始化为0
        //初始化old_sum的tile
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i);
//            printf("before copy in , %d\n", i);
            TLOAD(dataTile, gI);
            reduceprod_row_kernel<tile_shapeData, tile_shapeProd><<<tile_shapeProd::ValidRow, 1, 1>>>(ProdTile.data(), dataTile.data(), oldProdTile.data());
//            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//            printf("kernel , %d\n", i);
            oldProdTile = ProdTile;
        }
//        printf("end for%d\n",j);
        //for row corner
        if constexpr (rmd_N > 0){
            auto gI = gIIter(j, Nb);
            TLOAD(dataTile_row, gI);
            reduceprod_row_kernel<tile_shapeData_row, tile_shapeProd><<<tile_shapeProd::ValidRow, 1, 1>>>(ProdTile.data(), dataTile_row.data(), oldProdTile.data());
//            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
            oldProdTile = ProdTile;
        }
//        printf("before tstore\n");
        TSTORE(gO, ProdTile);
//        printf("end tstore\n");
    }
    //for col cor
    if constexpr (rmd_M > 0){
        auto gO = gOIter(Mb, 0);
        TEXPANDSCALAR(oldProdTile_col, 0);//初始化为0
        //初始化old_sum的tile
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);
            TLOAD(dataTile_col, gI);
            reduceprod_row_kernel<tile_shapeData_col, tile_shapeProd_col><<<tile_shapeProd_col::ValidRow, 1, 1>>>(ProdTile_col.data(), dataTile_col.data(), oldProdTile_col.data());
            oldProdTile_col = ProdTile_col;
        }
        if constexpr (rmd_N > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reduceprod_row_kernel<tile_shapeData_cor, tile_shapeProd_col><<<tile_shapeProd_col::ValidRow, 1, 1>>>(ProdTile_col.data(), dataTile_cor.data(), oldProdTile_col.data());
            oldProdTile_col = ProdTile_col;
        }
        TSTORE(gO, ProdTile_col);
    }
/*
    for(int i = 0; i < gIM; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
//    printf("end program\n");
}

#endif
