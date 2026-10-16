#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "solution/qli/qli_pto_opt_simple.hpp"

// 输入地址由链接器解析，修改 kernel 大小后无需重新修补常量地址。
extern "C" {
extern unsigned char _binary_srcq_data_start[];
extern unsigned char _binary_srck_data_start[];
extern unsigned char _binary_srcw_data_start[];
extern unsigned char _binary_srcsq_data_start[];
extern unsigned char _binary_srcsk_data_start[];
}

#ifdef QLI_INPUT_DATA
extern "C" uint32_t qli_workspace[];
#define OUT_SCORES reinterpret_cast<uintptr_t>(qli_workspace)
#else
#define OUT_SCORES 0x4000802000ULL
#endif
// indices 紧随 scores 之后，避免大 Sq*Skv 时与 scores 区域重叠
#define OUT_INDICES (OUT_SCORES + (uint64_t)Sq * Skv * 4)

// 尊重 Makefile -DBatch=$(B)；未定义时默认 1
#ifndef Batch
#define Batch 1
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

static_assert(Batch == 1, "original QLI driver supports exactly one batch");
static_assert(Sq > 0 && Skv > 0 && g > 0 && kTm > 0 && kTk > 0,
              "QLI dimensions must be positive");
static_assert(Skv % kTk == 0,
              "Skv must be divisible by the key tile width");
static_assert(g % kTm == 0 && kTm <= 32,
              "g must be divisible by kTm and kTm must be at most 32");
static_assert(topK > 0 && topK <= Skv,
              "topK must be in the range 1..Skv");

int main(){
    using dtype = DTYPE;

    dtype*   q       = reinterpret_cast<dtype*>(_binary_srcq_data_start);
    dtype*   k       = reinterpret_cast<dtype*>(_binary_srck_data_start);
    float*   w       = reinterpret_cast<float*>(_binary_srcw_data_start);
    float*   scale_q = reinterpret_cast<float*>(_binary_srcsq_data_start);
    float*   scale_k = reinterpret_cast<float*>(_binary_srcsk_data_start);

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
    for(int i=0;i<Batch;i++){
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
    for(int i=0;i<Batch;i++){
        qli_topk_radix<Sq, Skv, topK>(
            reinterpret_cast<float*>(OUT_SCORES) + i*Sq*Skv,
            reinterpret_cast<int32_t*>(OUT_INDICES) + i*Sq*topK
        );
    }
    __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

    return 0;
}
