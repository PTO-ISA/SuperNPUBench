#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define kM 1
#define kN 4096

#define kTM 1
#define kTN 4096

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    __bf16 *src;
    __bf16 *dst;

    // uint64_t p;
    // p = (uint64_t)malloc(kM*kN*sizeof(__bf16)+2*ALIGN);
    // src = (__bf16*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(kM*kN*sizeof(__bf16)+2*ALIGN);
    // dst = (__bf16*)(p & ALIGN_MASK);

    __bf16 srcp[kM*kN + 2*ALIGN];
    __bf16 dstp[kM*kN + 2*ALIGN];
    src = (__bf16*)(((uint64_t)srcp & ALIGN_MASK) + ALIGN);
    dst = (__bf16*)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    #ifdef OPT
    softmax_oneline<__bf16, kN>(dst, src);
    #else
    softmax_bf16<kM, kN, kTM, kTN>(dst, src);
    #endif
    BENCHEND;
}