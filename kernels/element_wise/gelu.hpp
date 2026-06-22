#include <common/pto_tileop.hpp>
#include "../test/accelerator/include/accelerator_fusion.h"

#include "template_asm.h"
#ifdef __linx
#include <stddef.h>
#include <stdint.h>
#else
#include <cstdint>
#include <cstdio>
#endif
// 海思解决方案 新版多项式拟合

// ==============================================
// ==============================================

template<typename tile_shape>
void __vec__ gelu_simd(
    typename tile_shape::TileDType __in__ in,
    typename tile_shape::TileDType __out__ out
    // bool approximate = false // false:none, true:tanh
) {
    size_t index = blkv_get_index_x();
    typename tile_shape::DType indata;
    indata.data = blkv_get_tile_ptr(in)[index].data; // 直接拷贝 short

    // 数据格式转换 V.FCVT
    float x = static_cast<float>(indata);
    
    constexpr uint32_t TOTAL_COUNT = 24*8*1024;
    constexpr float SCALAR_A5 = -3.5123395303315874e-09f;
    constexpr float SCALAR_A4 =  2.6452661927578447e-07f;
    constexpr float SCALAR_A3 = -7.9294877650681883e-06f;
    constexpr float SCALAR_A2 =  1.1061238183174282e-04f;
    constexpr float SCALAR_A1 =  6.5189960878342390e-05f;
    constexpr float SCALAR_A0 = -7.2666168212890625e-02f;
    constexpr float SCALAR_AM1 = -1.5957698822021484e+00f;
    constexpr float FP32_MAX = 5.75f;
    
    float t = blkv_max(x, -FP32_MAX);
    t = blkv_min(t, FP32_MAX);
    float t2 = t * t;

    float p = SCALAR_A5 * t2 + SCALAR_A4;
    p = p * t2 + SCALAR_A3;
    p = p * t2 + SCALAR_A2;
    p = p * t2 + SCALAR_A1;
    p = p * t2 + SCALAR_A0;
    p = p * t2 + SCALAR_AM1;

    float exp_val = blkv_fexp(t * p);
    float y = x / (1.0f + exp_val);
    
    BLKC_ASSIGN_CAST(out, index, y);
    // blkv_get_tile_ptr(out)[index] = static_cast<typename tile_shape::DType>(result);
}

template<typename tile_shapeData>
void gelu_impl(
    tile_shapeData &in,
    tile_shapeData &out
){
    static_assert(tile_shapeData::ValidRow != -1 && tile_shapeData::ValidCol != -1,
                  "Only static shape supported");
    gelu_simd<tile_shapeData><<<tile_shapeData::ValidCol, tile_shapeData::ValidRow, 1>>>(in.data(), out.data());
}

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
                printf("%.4f ", DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)

template<typename dtype, int gM, int tM>
void gelu(
    dtype *in_ptr,
    dtype *out_ptr,
    bool approximate = false // false:none, true:tanh
    ) {
    const int Mb = gM / tM;
    
    const int rmd_M = gM % tM;

    using gm_shape = global_tensor<dtype, RowMajor<1, gM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, rmd_M>;

    gm_shape inGm(in_ptr);
    gm_shape outGm(out_ptr);
    tile_shapeData inTile;
    tile_shapeData outTile;
    tile_shapeData_rmd inTile_rmd;
    tile_shapeData_rmd outTile_rmd;

    using itIn = global_iterator<gm_shape, tile_shapeData>;
    using itOut = global_iterator<gm_shape, tile_shapeData>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);
    // for test ///////////////////////////////////////
    alignas(256) static dtype  g_dump_intTile[tM];
    alignas(256) static dtype  g_dump_outTile[tM];
    // ///////////////////////////////////////

    // printf("MB = %d\n", Mb);
    for (int i = 0; i < Mb; ++i) {
        // printf("iter i %d\n",i);
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);
        TCOPYIN(inTile, gI);
        // DUMP_TILE("inTile", inTile, g_dump_intTile, 1, tM);
        gelu_impl<tile_shapeData>(inTile, outTile);
        // DUMP_TILE("outTile", outTile, g_dump_outTile, 1, tM);
        TCOPYOUT(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        auto gO = gOIter(0, Mb);
        TCOPYIN(inTile_rmd, gI);
        gelu_impl<tile_shapeData_rmd>(inTile_rmd, outTile_rmd);
        // DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tM);
        TCOPYOUT(gO, outTile_rmd);
    }
}
