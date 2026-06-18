#include <common/pto_tileop.hpp>
#include "../../include/accelerator_vec_simd.h"
#include "benchmark.h"
#include "common.h"

#define kM 1
#define kN 12288

#define kTM 1
#define kTN 1024

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 256

int main(){
    __bf16 *x;
    __bf16 *res;

    float *gamma;
    float *beta;

    // uint64_t p;
    // p = (uint64_t)malloc(kM*kN*sizeof(__bf16)+2*ALIGN);
    // x = (__bf16*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(kM*kN*sizeof(__bf16)+2*ALIGN);
    // res = (__bf16*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(1*sizeof(float)+2*ALIGN);
    // gamma= (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(1*sizeof(float)+2*ALIGN);
    // beta= (float*)(p & ALIGN_MASK);

    __bf16 xp[kM*kN+2*ALIGN];
    __bf16 resp[kM*kN+2*ALIGN];

    float gammap;
    float betap;
    x = (__bf16*)(((uint64_t)xp & ALIGN_MASK)+ALIGN);
    res = (__bf16*)(((uint64_t)resp & ALIGN_MASK)+ALIGN);
    gamma = &gammap;
    beta = &betap;

    BENCHSTART;
    layernorm_bf16<kM, kN, kTM, kTN>(res, x, gamma, beta);
    BENCHEND;
}