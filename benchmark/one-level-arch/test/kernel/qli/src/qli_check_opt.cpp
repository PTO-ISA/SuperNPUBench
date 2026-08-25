#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "qli/qli_pto_opt.hpp"

// P1 optimization: eliminate copy_bytes — pass .data segment absolute
// addresses directly to the kernel. The kernel's global_tensor + TLOAD
// can read from any mapped address, avoiding the byte-by-byte scalar copy
// that generated ~1.14M STD blocks (92.8% of total).
//
// Addresses updated via llvm-nm after each relink (same as qli_check.cpp).
#define SRCQ_ADDR  0x0000000000014670ULL
#define SRCK_ADDR  0x0000000000094670ULL
#define SRCW_ADDR  0x0000000000098670ULL
#define SRCSQ_ADDR  0x000000000009c670ULL
#define SRCSK_ADDR  0x00000000000a0670ULL

#define OUT_SCORES  0x4000802000ULL
// indices 紧随 scores 之后，避免大 Sq*Skv 时与 scores 区域重叠
#define OUT_INDICES (0x4000802000ULL + (uint64_t)Sq * Skv * 4)

// 尊重 Makefile -DBatch=$(B)；未定义时默认 1
#ifndef B
#define B 1
#endif

#ifndef Tsq
#define Sq 64
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 128
#else
#define Skv Tskv
#endif

#ifndef Tg
#define g 64
#else
#define g Tg
#endif

#define D 128

#ifndef Tm
#define kTm 16
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 32
#else
#define kTk Tk
#endif

#ifndef Ttopk
#define topK 8
#else
#define topK Ttopk
#endif

#ifdef QLI_INT8
#define DTYPE int8_t
#else
#define DTYPE __fp8_e4m3
#endif

int main(){
    using dtype = DTYPE;

    // P1: No copy_bytes — pass .data segment addresses directly.
    // The kernel's global_tensor accepts raw pointers; TLOAD reads from
    // the mapped .data region without needing aligned stack buffers.
    dtype*   q       = reinterpret_cast<dtype*>(SRCQ_ADDR);
    dtype*   k       = reinterpret_cast<dtype*>(SRCK_ADDR);
    float*   w       = reinterpret_cast<float*>(SRCW_ADDR);
    float*   scale_q = reinterpret_cast<float*>(SRCSQ_ADDR);
    float*   scale_k = reinterpret_cast<float*>(SRCSK_ADDR);

    BENCHSTART;
    for(int i=0;i<B;i++){
        qli_pto<dtype, Sq, Skv, D, g, kTm, kTk>(
            reinterpret_cast<float*>(OUT_SCORES) + i*Sq*Skv,
            q + i*Sq*g*D,
            k + i*Skv*D,
            w + i*Sq*g,
            scale_q + i*Sq*g,
            scale_k + i*Skv
        );
    }
    BENCHEND;

    // Step7 (TopK) 独立计时区间
    __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);
    for(int i=0;i<B;i++){
        qli_topk_radix<Sq, Skv, topK>(
            reinterpret_cast<float*>(OUT_SCORES) + i*Sq*Skv,
            reinterpret_cast<int32_t*>(OUT_INDICES) + i*Sq*topK
        );
    }
    __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

    return 0;
}
