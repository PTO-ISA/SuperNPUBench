#ifndef TRANSPOSE_KERNEL_HPP
#define TRANSPOSE_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include "template_asm.h"

using namespace pto;

#pragma once
#include <cstdint>
#include <cstdio>


#define DUMP_TILE(label, TileVar, DumpBuf, Rows, Cols) \
    do { \
        GlobalTensor<typename decltype(TileVar)::DType, \
                     Shape<1,1,1,Rows,Cols>, \
                     Stride<1,1,1,Cols,1>> _g(DumpBuf); \
        TCOPYOUT(_g, TileVar); \
        printf("[DUMP] %s (shape=%dx%d):\n", label, Rows, Cols); \
        for (int ri = 0; ri < Rows; ri++) { \
            printf("  row%2d: ", ri); \
            for (int ci = 0; ci < Cols; ci++) \
                printf("%12lld ", (long long)DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)

// ==============================================
// 维度规则：交换transpose_dim0和transpose_dim1
// ==============================================
template<typename dtype, typename tile_shape, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t TRANSPOSE_DIM1, size_t TRANSPOSE_DIM0>
void __vec__ gen_offset_trans(
    typename tile_shape::TileDType __out__ out,
    const size_t *in_shape,
    const size_t *out_shape,
//    const size_t in_dim,
//    const size_t out_dim,
//    const size_t transpose_dim1,
//    const size_t transpose_dim0,    
    const size_t base,
    const size_t total_elements
) {
    size_t index = blkv_get_index_x();
    size_t idx = blkv_get_index_x();
    if (index >= total_elements) return;
    idx = idx + base;   // todo idx是个向量，base是个标量，获得所有的基地址或者说基offset

    //转置维度交换stride。
//    uint16_t stride[IN_DIM];
    uint32_t stride[IN_DIM];
    stride[IN_DIM-1] = 1;

    #pragma clang loop unroll(full)
    for(int i = IN_DIM-2 ; i >= 0; --i)
    {
        stride[i] = stride[i+1] * in_shape[i+1];
    }
    std::swap(stride[TRANSPOSE_DIM1],stride[TRANSPOSE_DIM0]);

    // 输出一维索引 → 输出坐标
    size_t out_coord[MAX_DIM] = {0};    //
    size_t tmp = idx;   //
    
    #pragma clang loop unroll(full)
    for (int d = OUT_DIM - 1; d >= 0; d--) {
        out_coord[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

/*
    // 输出坐标 → 输入坐标
    size_t in_coord[MAX_DIM] = {0};
    for (size_t i = 0; i < in_dim; i++) {
        size_t o = out_dim - in_dim + i;  // 从后面对齐
        if (in_shape[i] == 1) {
            in_coord[i] = 0;
        } else {
            in_coord[i] = out_coord[o];
        }
    }
*/
//    uint16_t in_offset = 0; 
    uint32_t in_offset = 0;   

    #pragma clang loop unroll(full)    
    for (int i = 0; i < IN_DIM; i++) {
        in_offset += out_coord[i] * stride[i] * sizeof(dtype);
    }

    // 赋值
    blkv_get_tile_ptr(out)[index] = in_offset;
}

template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t TRANSPOSE_DIM1, size_t TRANSPOSE_DIM0>
void gen_offset_Impl(
    tile_shapeOffset &out,
    const size_t *in_shape,
    const size_t *out_shape,
//    const size_t in_dim,
//   const size_t out_dim,
//    const size_t transpose_dim1,
//    const size_t transpose_dim0,   
    const size_t base,
    const size_t total_elements
    )
{
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");
    gen_offset_trans<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM, TRANSPOSE_DIM1, TRANSPOSE_DIM0><<<tile_shapeOffset::ValidCol, tile_shapeOffset::ValidRow, 1>>>(out.data(), in_shape, out_shape, base, total_elements);  // todo 这部分的tile shape是怎么传递的？
}



template<typename dtype, size_t MAX_DIM = 8, const int gIM, const int gOM, const int tM, size_t IN_DIM, size_t OUT_DIM, size_t TRANSPOSE_DIM1, size_t TRANSPOSE_DIM0>
void transpose(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape
//    const size_t in_dim,
//    const size_t out_dim,
//    const size_t transpose_dim1,
//    const size_t transpose_dim0   
) 
    {

    const int Mb = gOM / tM;

    
    const int rmd_M = gOM % tM; // todo 尾块怎么处理？

    using gm_shapeIn = global_tensor<dtype, RowMajor<1, gIM>>;     //将gm中的Tensor先声明为一维数据 
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeOffset = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec
//    using tile_shapeOffset = Tile<Location::Vec, uint16_t, 1, tM, BLayout::RowMajor>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec

    gm_shapeIn inGm(in_ptr);
    gm_shapeIn offGm(in_ptr);
    tile_shapeData dataTile;
    tile_shapeOffset offsetTile;

    int base = 0;// todo 生成一个标量
    int all_num = gOM; // 总元素数量

    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;

    itOut gOIter(out_ptr);
    alignas(256) static uint32_t  g_dump[tM];

    int total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);
        gen_offset_Impl<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM, TRANSPOSE_DIM1, TRANSPOSE_DIM0>(offsetTile, in_shape, out_shape, base, total_elements);
//        printf("end genoffset\n");
        base += total_elements;
//        DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tM);
        MGATHER(dataTile, inGm, offsetTile);
//        printf("end mgather\n");        
        TCOPYOUT(gO, dataTile);
//        TCOPYOUT(gO, dataTile);
    }
    if constexpr (rmd_M) {
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;//尾片的大小。
        gen_offset_Impl<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM, TRANSPOSE_DIM1, TRANSPOSE_DIM0>(offsetTile, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(dataTile, inGm, offsetTile);
        TCOPYOUT(gO, dataTile);
    }
}

#endif
