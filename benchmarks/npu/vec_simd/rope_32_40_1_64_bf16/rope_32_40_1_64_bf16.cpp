#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define row 1024
#define rope_dim 64

#define trow 64
#define tcol 32

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    __bf16 *x;
    __bf16 *freqs_cis;
    __bf16 *out;

    // uint64_t p;
    // p = (uint64_t)malloc(row*rope_dim*sizeof(__bf16)+2*ALIGN);
    // x = (__bf16*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(row*rope_dim*sizeof(__bf16)+2*ALIGN);
    // freqs_cis = (__bf16*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(row*rope_dim*sizeof(__bf16)+2*ALIGN);
    // out = (__bf16*)(p & ALIGN_MASK);

    __bf16 xp[row*rope_dim+2*ALIGN];
    __bf16 freqs_cisp[row*rope_dim+2*ALIGN];
    __bf16 outp[row*rope_dim+2*ALIGN];

    x = (__bf16*)(((uint64_t)xp & ALIGN_MASK) + ALIGN);
    freqs_cis = (__bf16*)(((uint64_t)freqs_cisp & ALIGN_MASK) + ALIGN);
    out = (__bf16*)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    rope<row, rope_dim, trow, tcol>(out, x, freqs_cis);
    BENCHEND;
}