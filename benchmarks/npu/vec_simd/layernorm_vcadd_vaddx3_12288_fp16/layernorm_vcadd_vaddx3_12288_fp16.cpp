#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define kM 1
#define kN 12288

#define kTM 1
#define kTN 1024

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    __half *x;
    __half *res;

    float *gamma;
    float *beta;

    // uint64_t p;
    // p = (uint64_t)malloc(kM*kN*sizeof(__half)+2*ALIGN);
    // x = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(kM*kN*sizeof(__half)+2*ALIGN);
    // res = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(1*sizeof(float)+2*ALIGN);
    // gamma= (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(1*sizeof(float)+2*ALIGN);
    // beta= (float*)(p & ALIGN_MASK);

    __half xp[kM*kN+2*ALIGN];
    __half resp[kM*kN+2*ALIGN];

    float gammap;
    float betap;
    x = (__half*)(((uint64_t)xp & ALIGN_MASK)+ALIGN);
    res = (__half*)(((uint64_t)resp & ALIGN_MASK)+ALIGN);
    gamma = &gammap;
    beta = &betap;

    BENCHSTART;
    layernorm<__half, kM, kN, kTM, kTN>(res, x, gamma, beta);
    BENCHEND;
}