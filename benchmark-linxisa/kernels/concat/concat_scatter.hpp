#ifndef CONCAT_SCATTER_KERNEL_HPP
#define CONCAT_SCATTER_KERNEL_HPP

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
template<typename dtype, typename tile_shape, typename tile_Inshape, typename tile_Outshape, size_t MAX_DIM = 8, size_t DATA_DIM, size_t CONCAT_DIM>
void __vec__ gen_offset_concat(
    typename tile_shape::TileDType __out__ out,
    typename tile_Inshape::TileDType __in__ in_shape,
    typename tile_Outshape::TileDType  __in__ out_shape,
//    const size_t in_dim, 
    const size_t base,
    const size_t total_elements
) {
    size_t index = blkv_get_index_x();
    size_t idx = blkv_get_index_x();

    __vbuf__ typename tile_Inshape::DType *in_shape_ptr = blkv_get_tile_ptr(in_shape);  
    __vbuf__ typename tile_Outshape::DType *out_shape_ptr = blkv_get_tile_ptr(out_shape);      

    if (index >= total_elements) return;
    idx = idx + base;   // todo idx是个向量，base是个标量，获得所有的基地址或者说基offset

    //转置维度交换stride。
//    uint16_t stride[IN_DIM];
    uint16_t stride[DATA_DIM];
    stride[DATA_DIM-1] = 1;

    #pragma clang loop unroll(full)
    for(int i = DATA_DIM-2 ; i >= 0; --i)
    {
        stride[i] = stride[i+1] * out_shape_ptr[i+1];
    }


    // 输出一维索引 → 输出坐标
    size_t in_coord[MAX_DIM] = {0};    //
    size_t tmp = idx;   //
    
    #pragma clang loop unroll(full)
    for (int d = DATA_DIM - 1; d >= 0; d--) {
        in_coord[d] = tmp % in_shape_ptr[d];
        tmp /= in_shape_ptr[d];
    }
    size_t n = tmp;
    in_coord[CONCAT_DIM] = n * in_shape_ptr[CONCAT_DIM] + in_coord[CONCAT_DIM];

//    size_t n = out_coord[CONCAT_DIM] / in_shape_ptr[CONCAT_DIM]; 
//    size_t offset = out_coord[CONCAT_DIM] % in_shape_ptr[CONCAT_DIM];

//    out_coord[CONCAT_DIM] = offset;

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
//    uint32_t out_offset = 0;  
    uint16_t out_offset = 0;        

    #pragma clang loop unroll(full)    
    for (int i = 0; i < DATA_DIM; i++) {
        out_offset += in_coord[i] * stride[i] * sizeof(dtype);
    }

    // 赋值
    blkv_get_tile_ptr(out)[index] = out_offset;
}

template<typename dtype, typename tile_shapeOffset, typename tile_Inshape, typename tile_Outshape, size_t MAX_DIM = 8, size_t DATA_DIM, size_t CONCAT_DIM>
void gen_offset_Impl(
    tile_shapeOffset &out,
    tile_Inshape &in_shape,
    tile_Outshape &out_shape,
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
    gen_offset_concat<dtype, tile_shapeOffset, tile_Inshape, tile_Outshape, MAX_DIM, DATA_DIM, CONCAT_DIM><<<tile_shapeOffset::ValidCol, tile_shapeOffset::ValidRow, 1>>>(out.data(), in_shape.data(), out_shape.data(), base, total_elements);  // todo 这部分的tile shape是怎么传递的？
}



template<typename dtype, size_t MAX_DIM = 8, const int gIM, const int gOM, const int tM, size_t DATA_DIM, size_t CONCAT_DIM>
void concat_scatter(
    dtype *in_ptr,
    dtype *out_ptr,
    size_t *in_shape,
    size_t *out_shape
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

    using gm_InDataShape = global_tensor<size_t, RowMajor<1, DATA_DIM>>;     //将gm中的Tensor先声明为一维数据 
    using gm_OutDataShape = global_tensor<size_t, RowMajor<1, DATA_DIM>>;

    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>; // todo 尾块怎么处理？是否要作为参数写在这
    using tile_shapeOffset = Tile<Location::Vec, uint16_t, 1, tM, BLayout::RowMajor>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec    
//    using tile_shapeOffset = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec

    using tile_Inshape = Tile<Location::Vec, size_t, 1, 32, BLayout::RowMajor, 1, DATA_DIM>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec
    using tile_Outshape = Tile<Location::Vec, size_t, 1, 32, BLayout::RowMajor, 1, DATA_DIM>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec        
//    using tile_shapeOffset = Tile<Location::Vec, uint16_t, 1, tM, BLayout::RowMajor>; // todo 这里的location，一定要是Vec吗？哪怕没有传入Vec

//    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    gm_InDataShape InShapeGm(in_shape);   
    gm_OutDataShape OutShapeGm(out_shape);      

    tile_shapeData dataTile;
    tile_shapeOffset offsetTile;

    tile_Inshape InshapeTile;
    tile_Outshape OutshapeTile;

    int base = 0;// todo 生成一个标量
    int all_num = gIM; // 总元素数量

//    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;

    itIn  gIIter(in_ptr);
//    itOut gOIter(out_ptr);

    alignas(256) static uint16_t  g_dump[tM];

    int total_elements = tM;


    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);        
        TCOPYIN(InshapeTile, InShapeGm);
        TCOPYIN(OutshapeTile, OutShapeGm);
        gen_offset_Impl<dtype, tile_shapeOffset, tile_Inshape, tile_Outshape, MAX_DIM, DATA_DIM, CONCAT_DIM>(offsetTile, InshapeTile, OutshapeTile, base, total_elements);
//        printf("end genoffset\n");
        base += total_elements;
//        DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tM);
        TCOPYIN(dataTile, gI);
        MSCATTER(outGm, dataTile, offsetTile);
    }
    if constexpr (rmd_M) {  
        auto gI = gIIter(0, Mb);                  
        TCOPYIN(InshapeTile, InShapeGm);
        TCOPYIN(OutshapeTile, OutShapeGm);        
        total_elements = rmd_M;//尾片的大小。
        gen_offset_Impl<dtype, tile_shapeOffset, tile_Inshape, tile_Outshape, MAX_DIM, DATA_DIM, CONCAT_DIM>(offsetTile, InshapeTile, OutshapeTile, base, total_elements);
        base += total_elements;
        TCOPYIN(dataTile, gI);
        MSCATTER(outGm, dataTile, offsetTile);
    }
}

#endif

