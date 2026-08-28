#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "qli/qli_pto_opt_dynamic.hpp"

// 动态 shape + 多 PE 版 driver（地址由 fix_cpp_addrs.py 迭代更新）
#define SRCQ_ADDR  0x0000000000014f60ULL
#define SRCK_ADDR  0x0000000000094f60ULL
#define SRCW_ADDR  0x0000000000098f60ULL
#define SRCSQ_ADDR  0x000000000009cf60ULL
#define SRCSK_ADDR  0x00000000000a0f60ULL

#define OUT_SCORES  0x4000802000ULL
// scores 行步长为 paddedSkv = ceil(Skv/2048)*2048（TopK 整块读取契约）
#define PADDED_SKV (((Skv + 2047) / 2048) * 2048)
#define OUT_INDICES (OUT_SCORES + (uint64_t)Sq * PADDED_SKV * 4)
// key/hist/prefix scratch 紧随 indices 之后
#define KEY_SCRATCH (OUT_INDICES + (uint64_t)Sq * topK * 4 + 8192)

#ifndef B
#define B 1
#endif
#ifndef Tsq
#define Sq 4
#else
#define Sq Tsq
#endif
#ifndef Tskv
#define Skv 128
#else
#define Skv Tskv
#endif
#ifndef Ttopk
#define topK 8
#else
#define topK Ttopk
#endif
#ifndef TNPE
#define numPEs 1
#else
#define numPEs TNPE
#endif

#define DTYPE __fp8_e4m3

int main() {
    using dtype = DTYPE;
    dtype*   q       = reinterpret_cast<dtype*>(SRCQ_ADDR);
    dtype*   k       = reinterpret_cast<dtype*>(SRCK_ADDR);
    float*   w       = reinterpret_cast<float*>(SRCW_ADDR);
    float*   scale_q = reinterpret_cast<float*>(SRCSQ_ADDR);
    float*   scale_k = reinterpret_cast<float*>(SRCSK_ADDR);
    float*   scores  = reinterpret_cast<float*>(OUT_SCORES);
    int32_t* indices = reinterpret_cast<int32_t*>(OUT_INDICES);
    uint32_t* key_scratch = reinterpret_cast<uint32_t*>(KEY_SCRATCH);

    BENCHSTART;
    for (int i = 0; i < B; i++) {
        qli_pto_dynamic<dtype>(
            scores + i * Sq * PADDED_SKV,
            q + i * Sq * 64 * 128, k,
            w + i * Sq * 64, scale_q + i * Sq * 64, scale_k,
            Sq, Skv, numPEs);
    }
    BENCHEND;

    __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);
    for (int i = 0; i < B; i++) {
        qli_topk_radix_dynamic(
            scores + i * Sq * PADDED_SKV,
            indices + i * Sq * topK,
            Sq, Skv, topK, numPEs,
            key_scratch + i * Sq * PADDED_SKV);
    }
    __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

    return 0;
}
