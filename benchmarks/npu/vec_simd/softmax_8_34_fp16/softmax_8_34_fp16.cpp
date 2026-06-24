#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define kM 8
#define kN 34

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

void softmax_local(__half* dst, __half* src){
    using gm_shape = global_tensor<__half, RowMajor<8, 34>>;
    using tile_shape = Tile<Location::Vec, __half, 8, 64, BLayout::RowMajor, 8, 34>;

    using tMax = Tile<Location::Vec, __half, 8, 64, BLayout::RowMajor, 8, 1>;
    using tSum = Tile<Location::Vec, __half, 8, 64, BLayout::RowMajor, 8, 1>;

    gm_shape gsrc(src);
    tile_shape tsrc;
    TCOPYIN(tsrc, gsrc);

    tMax tLocalMax;
    TROWMAX(tLocalMax, tsrc);

    tile_shape tExp;
    TEXPANDCOL(tExp, tLocalMax);
    TSUB(tExp, tsrc, tExp);

    Tile<Location::Vec, float, 8, 64, BLayout::RowMajor, 8, 34> tcast;
    TCAST(tcast, tExp);
    TEXP(tcast, tcast);
    TCAST(tExp, tcast);

    tSum tLocalSum;
    TROWSUM(tLocalSum, tExp);

    tile_shape tSumExpand;
    TEXPANDCOL(tSumExpand, tLocalSum);

    tile_shape tres;
    TDIV(tres, tExp, tSumExpand);

    gm_shape gdst(dst);
    TCOPYOUT(gdst, tres);
}

int main(){
    __half *src;
    __half *dst;

    // uint64_t p;
    // p = (uint64_t)malloc(8*34*sizeof(__half)+2*ALIGN);
    // src = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(8*34*sizeof(__half)+2*ALIGN);
    // dst = (__half*)(p & ALIGN_MASK);

    __half srcp[8*34 + 2*ALIGN];
    __half dstp[8*34 + 2*ALIGN];
    src = (__half*)(((uint64_t)srcp & ALIGN_MASK) + ALIGN);
    dst = (__half*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    softmax_local(dst, src);
    BENCHEND;
}