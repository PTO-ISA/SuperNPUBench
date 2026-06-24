#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simd.h>
#include "benchmark.h"
#include "common.h"

#define M 18
#define N 256
#define K 128

#define tM 64
#define tN 64
#define tK 64

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    float alpha;
    float beta;

    float *a;
    float *b;
    float *c;

    // uint64_t p;
    // p = (uint64_t)malloc(M*K*sizeof(float)+2*ALIGN);
    // a = (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(K*N*sizeof(float)+2*ALIGN);
    // b = (float*)(p & ALIGN_MASK);

    // p = (uint64_t)malloc(M*N*sizeof(float)+2*ALIGN);
    // c = (float*)(p & ALIGN_MASK);

    float ap[M*K+2*ALIGN];
    float bp[K*N+2*ALIGN];
    float cp[M*N+2*ALIGN];

    a = (float*)(((uint64_t)ap & ALIGN_MASK)+ALIGN);
    b = (float*)(((uint64_t)bp & ALIGN_MASK)+ALIGN);
    c = (float*)(((uint64_t)cp & ALIGN_MASK)+ALIGN);

    BENCHSTART;
    gemm<M, N, K, tM, tN, tK, true>(c, a, b, alpha, beta);
    BENCHEND;
}