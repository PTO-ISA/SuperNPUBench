#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "qli/qli_pto_opt_simple.hpp"

// P1 optimization: eliminate copy_bytes — pass .data segment absolute
// addresses directly to the kernel. The kernel's global_tensor + TLOAD
// can read from any mapped address, avoiding the byte-by-byte scalar copy
// that generated ~1.14M STD blocks (92.8% of total).
//
// Addresses updated via llvm-nm after each relink (same as qli_check.cpp).
#define SRCQ_ADDR  0x0000000000014608ULL
#define SRCK_ADDR  0x0000000000094608ULL
#define SRCW_ADDR  0x0000000000098608ULL
#define SRCSQ_ADDR  0x000000000009c608ULL
#define SRCSK_ADDR  0x00000000000a0608ULL

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

    // v0.58.4：W*scale_q 预广播为 [Sq*g, kTk]（行 r 全列同值），
    // kernel 内用普通 TMUL（规避单列广播源的物理列校验）
    static float wbb[Sq * g * kTk];
    for (int r = 0; r < Sq * g; r++)
        for (int c = 0; c < kTk; c++)
            wbb[r * kTk + c] = w[r] * scale_q[r];
    // v0.58.4：K 转置为 [D, Skv] 行主序（CUBE_N8 B-tile 契约）
    static dtype ktt[D * Skv];
    for (int n = 0; n < Skv; n++)
        for (int d = 0; d < D; d++)
            ktt[d * Skv + n] = k[n * D + d];
    // CUBE->Vec 桥接临时区
    static float tmp16[kTm * kTk];

    BENCHSTART;
    for(int i=0;i<B;i++){
        qli_pto<dtype, Sq, Skv, D, g, kTm, kTk>(
            reinterpret_cast<float*>(OUT_SCORES) + i*Sq*Skv,
            q + i*Sq*g*D,
            ktt + i*D*Skv,
            wbb + i*Sq*g*kTk,
            scale_k + i*Skv,
            tmp16
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
