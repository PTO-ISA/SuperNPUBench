#include <common/pto_tileop.hpp>
#include "../../include/accelerator_vec_simd.h"
#include "benchmark.h"
#include "common.h"

#define tokens 1024
#define scores 256

#define tS 8
#define tK 16

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 256

int main(){
    float *x;
    float *weight;
    float *indices;

    // uint64_t p;
    // p = (uint64_t)malloc(tokens*scores*sizeof(float)+2*ALIGN);
    // x = (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(tokens*tK*sizeof(float)+2*ALIGN);
    // weight = (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(tokens*tK*sizeof(float)+2*ALIGN);
    // indices = (float*)(p & ALIGN_MASK);

    float xp[tokens*scores];
    float weightp[tokens*tK];
    float indicesp[tokens*tK];

    x = (float*)(((uint64_t)xp & ALIGN_MASK) + ALIGN);
    weight = (float*)(((uint64_t)weightp & ALIGN_MASK) + ALIGN);
    indices = (float*)(((uint64_t)indicesp & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    topk<float, tokens, scores, tS, tK>(weight, indices, x);
    BENCHEND;
}