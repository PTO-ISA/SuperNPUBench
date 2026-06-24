#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define kM 1
#define kN 16384

#define kTM 1
#define kTN 16384

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    __half *src;
    __half *dst;
    
    // uint64_t p;
    // p = (uint64_t)malloc(kM*kN*sizeof(__half)+2*ALIGN);
    // src = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(kM*kN*sizeof(__half)+2*ALIGN);
    // dst = (__half*)(p & ALIGN_MASK);

    __half srcp[kM*kN + 2*ALIGN];
    __half dstp[kM*kN + 2*ALIGN];
    src = (__half*)(((uint64_t)srcp & ALIGN_MASK) + ALIGN);
    dst = (__half*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    #ifdef OPT
    softmax_oneline<__half, kN>(dst, src);
    #else
    softmax<__half, kM, kN, kTM, kTN>(dst, src);
    #endif
    BENCHEND;
}