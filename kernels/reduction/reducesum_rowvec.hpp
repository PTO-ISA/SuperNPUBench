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

template<typename tileSrc, typename tileSum>
void __vec__ reducesum_row_kernel(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileSrc::TileDType __in__ src,
    const typename tileSum::TileDType __in__ old_sum
)
{
//    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_x();
//    size_t j = blkv_get_index_y();
    size_t idx = j * tileSum::RowStride;

    __vbuf__ typename tileSum::DType *new_sum_ptr = blkv_get_tile_ptr(new_sum);
    __vbuf__ typename tileSrc::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tileSum::DType *old_sum_ptr = blkv_get_tile_ptr(old_sum);


    typename tileSum::DType upd_sum = old_sum_ptr[idx];

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

        typename tileSum::DType sum_01 = src_ptr[src_idx0] + src_ptr[src_idx1];
        typename tileSum::DType sum_23 = src_ptr[src_idx2] + src_ptr[src_idx3];
        typename tileSum::DType sum_45 = src_ptr[src_idx4] + src_ptr[src_idx5];
        typename tileSum::DType sum_67 = src_ptr[src_idx6] + src_ptr[src_idx7];

        typename tileSum::DType sum_0123 = sum_01 + sum_23;
        typename tileSum::DType sum_4567 = sum_45 + sum_67;

        typename tileSum::DType sum_tmp = sum_0123 + sum_4567;

        upd_sum = upd_sum + sum_tmp;
    }

/*
    for(size_t i=0;i<tileSrc::ValidCol;i++){
        size_t src_idx =  i * tileSrc::ColStride + j * tileSrc::RowStride;
        upd_sum = upd_sum + src_ptr[src_idx];
    }
*/
    new_sum_ptr[idx] = upd_sum;

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

    tile_shapeSum SumTile;
    tile_shapeSum oldSumTile;
    tile_shapeSum_col SumTile_col;
    tile_shapeSum_col oldSumTile_col;

//    int base = 0;// todo 生成一个标量
//    int all_num = gOM; // 总元素数量

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeSum>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

//    printf("tile_shapeSum::ValidCol = %d\n",  tile_shapeSum::ValidCol);
//    printf("tile_shapeSum::ValidRow = %d\n",  tile_shapeSum::ValidRow);
//    printf("before for\n");
    for (int j = 0; j < Mb; ++j) {
        auto gO = gOIter(j, 0);
        TEXPANDSCALAR(oldSumTile, 0);//初始化为0
        //初始化old_sum的tile
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(j, i);
//            printf("before copy in , %d\n", i);
            TLOAD(dataTile, gI);
            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//            reducesum_row_kernel<tile_shapeData, tile_shapeSum><<<1, tile_shapeSum::ValidRow, 1>>>(SumTile.data(), dataTile.data(), oldSumTile.data());
//            printf("kernel , %d\n", i);
            oldSumTile = SumTile;
        }
//        printf("end for%d\n",j);
        //for row corner
        if constexpr (rmd_N > 0){
            auto gI = gIIter(j, Nb);
            TLOAD(dataTile_row, gI);
            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
//            reducesum_row_kernel<tile_shapeData_row, tile_shapeSum><<<tile_shapeSum::ValidRow, 1, 1>>>(SumTile.data(), dataTile_row.data(), oldSumTile.data());
            oldSumTile = SumTile;
        }
//        printf("before tstore\n");
        TSTORE(gO, SumTile);
//        printf("end tstore\n");
    }
    //for col cor
    if constexpr (rmd_M > 0){
        auto gO = gOIter(Mb, 0);
        TEXPANDSCALAR(oldSumTile_col, 0);//初始化为0
        //初始化old_sum的tile
        for (int i = 0; i < Nb; ++i) {
            auto gI = gIIter(Mb, i);
            TLOAD(dataTile_col, gI);
            reducesum_row_kernel<tile_shapeData_col, tile_shapeSum_col><<<tile_shapeSum_col::ValidRow, 1, 1>>>(SumTile_col.data(), dataTile_col.data(), oldSumTile_col.data());
            oldSumTile_col = SumTile_col;
        }
        if constexpr (rmd_N > 0){
            auto gI = gIIter(Mb, Nb);
            TLOAD(dataTile_cor, gI);
            reducesum_row_kernel<tile_shapeData_cor, tile_shapeSum_col><<<tile_shapeSum_col::ValidRow, 1, 1>>>(SumTile_col.data(), dataTile_cor.data(), oldSumTile_col.data());
            oldSumTile_col = SumTile_col;
        }
        TSTORE(gO, SumTile_col);
    }
/*
    for(int i = 0; i < gIM; i++){
        printf("out%d = %d\n", i, out_ptr[i]);
    }
*/
//    printf("end program\n");
}

#endif
