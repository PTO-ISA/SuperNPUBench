#include <common/pto_tileop.hpp>
#include "../test/accelerator/include/accelerator_fusion.h"

#include "template_asm.h"
#include <cstdint>
#include <cstdio>
// 旧版多项式拟合&tanh近似

// 多项式累加拟合
// erf(3.92) = 0.9999999999 → 几乎 = 1
// erf(-3.92) = -0.9999999999 → 几乎 = -1
inline float my_erff(const float x) {
    constexpr float SCALAR_P0 = 0.29639384698e5;
    constexpr float SCALAR_P1 = 0.50637915060e4;
    constexpr float SCALAR_P2 = 0.13938061484e4;
    constexpr float SCALAR_P3 = 0.10162808918e3;
    constexpr float SCALAR_P4 = 0.75517016694e1;
    constexpr float SCALAR_P5 = 0.053443748819;
    constexpr float SCALAR_Q0 = 0.26267224157e5;
    constexpr float SCALAR_Q1 = 0.13243365831e5;
    constexpr float SCALAR_Q2 = 0.30231248150e4;
    constexpr float SCALAR_Q3 = 0.39856963806e3;
    constexpr float SCALAR_Q4 = 0.31212858877e2;
    constexpr float FP32_MIN = -3.92;
    constexpr float FP32_MAX = 3.92;
    float result;
    float result_Q;
    float x2;
    float xclip;
    // clip, 避免多项式计算溢出
    // if (x > FP32_MAX) {
    //     return result = 1.0f;
    // }
    // if (x < FP32_MIN) {
    //     return result = -1.0f;
    // }
    xclip = blkv_max(x, FP32_MIN);
    xclip = blkv_min(xclip, FP32_MAX);

    x2 = xclip * xclip;
    result = SCALAR_P5 * x2 + SCALAR_P4;
    result = result * x2 + SCALAR_P3;
    result = result * x2 + SCALAR_P2;
    result = result * x2 + SCALAR_P1;
    result = result * x2 + SCALAR_P0;
    result_Q = x2 + SCALAR_Q4;
    result_Q = result_Q * x2 + SCALAR_Q3;
    result_Q = result_Q * x2 + SCALAR_Q2;
    result_Q = result_Q * x2 + SCALAR_Q1;
    result_Q = result_Q * x2 + SCALAR_Q0;
    return (result * xclip / result_Q);
}

inline float my_tanh(const float x) {
    constexpr float FP32_MIN = -8.8;
    constexpr float FP32_MAX = 8.8;
    float xclip;
    float exp2x;
    // clip, 避免多项式计算溢出
    // if (x > FP32_MAX) {
    //     return result = 1.0f;
    // }
    // if (x < FP32_MIN) {
    //     return result = -1.0f;
    // }
    // todo 优化思路
    /*
    实际上，只要判断输入值超过clip范围，直接输出1
    但因为simd只有在整个向量所有元素都超过范围时，才能跳过指数计算，输出1，因此，这里的判断可能在大部分情况反而带来负收益
    */
    xclip = blkv_max(x, FP32_MIN);
    xclip = blkv_min(xclip, FP32_MAX);

    exp2x = blkv_fexp(xclip * 2.0f);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

template<typename tile_shape>
void __vec__ gelu_simd(
    typename tile_shape::TileDType __in__ in,
    typename tile_shape::TileDType __out__ out,
    bool approximate = false // false:none, true:tanh
) {
    size_t index = blkv_get_index_x();
    typename tile_shape::DType indata;
    indata.data = blkv_get_tile_ptr(in)[index].data; // 直接拷贝 short

    // size_t index = blkv_get_index_x();
    // typename tile_shape::DType indata  = blkv_get_tile_ptr(in)[index];

    float result;
    // 数据格式转换 V.FCVT
    float x = static_cast<float>(indata);
    
    // GELU(x)=x∗Φ(x), Φ(x)=负无穷~x积分 φ(exp(-0.5f*x*x) / sqrt(2π))
    // 等价于GELU(x)=0.5⋅x⋅(1+erf(x/sqrt(2))
    if (!approximate) {
        result = 0.5f * x * (1.0f + my_erff(x * 0.707107f));

    // GELU(x)=0.5∗x∗(1+Tanh(sqr(2/π) ∗ (x+0.044715∗x**3)))
    } else {
        // 计算
        constexpr float beta = 0.7978845608f; // sqrt(2/π)
        constexpr float kappa = 0.044715f;
        float x3 = x * x * x;
        float inner = beta * (x + kappa * x3);
        result = 0.5f * x * (1.0f + my_tanh(inner));
    }
    BLKC_ASSIGN_CAST(out, index, result);
    // blkv_get_tile_ptr(out)[index] = static_cast<typename tile_shape::DType>(result);
}

template<typename tile_shapeData>
void gelu_impl(
    tile_shapeData &in,
    tile_shapeData &out,
    bool approximate = false // false:none, true:tanh
){
    static_assert(tile_shapeData::ValidRow != -1 && tile_shapeData::ValidCol != -1,
                  "Only static shape supported");
    gelu_simd<tile_shapeData><<<tile_shapeData::ValidCol, tile_shapeData::ValidRow, 1>>>(in.data(), out.data(), approximate);
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
        gelu_impl<tile_shapeData>(inTile, outTile, approximate);
        // DUMP_TILE("outTile", outTile, g_dump_outTile, 1, tM);
        TCOPYOUT(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        auto gO = gOIter(0, Mb);
        TCOPYIN(inTile_rmd, gI);
        gelu_impl<tile_shapeData_rmd>(inTile_rmd, outTile_rmd, approximate);
        // DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tM);
        TCOPYOUT(gO, outTile_rmd);
    }

}