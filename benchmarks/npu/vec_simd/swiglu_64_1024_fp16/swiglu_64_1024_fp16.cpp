#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define S 64
#define InDim 1024
#define OutDim 1024

#define tS 64
#define tInDim 128
#define tOutDim 64

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    __half *in;
    __half *w1;
    __half *w2;
    __half *out;

    // uint64_t p;
    // p = (uint64_t)malloc(S*InDim*sizeof(__half)+2*ALIGN);
    // in = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(InDim*OutDim*sizeof(__half)+2*ALIGN);
    // w1 = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(InDim*OutDim*sizeof(__half)+2*ALIGN);
    // w2 = (__half*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(S*OutDim*sizeof(__half)+2*ALIGN);
    // out = (__half*)(p & ALIGN_MASK);

    __half inp[S*InDim + 2*ALIGN];
    __half w1p[InDim*OutDim + 2*ALIGN];
    __half w2p[InDim*OutDim + 2*ALIGN];
    __half outp[S*OutDim + 2*ALIGN];

    in = (__half*)(((uint64_t)inp & ALIGN_MASK) + ALIGN);
    w1 = (__half*)(((uint64_t)w1p & ALIGN_MASK) + ALIGN);
    w2 = (__half*)(((uint64_t)w2p & ALIGN_MASK) + ALIGN);
    out = (__half*)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    swiglu<__half, S, InDim, OutDim, tS, tInDim, tOutDim>(out, in, w1, w2);
    BENCHEND;
}