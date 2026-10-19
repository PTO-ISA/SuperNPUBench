#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "solution/qli/qli_pto_opt_dynamic.hpp"

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
// scores 行步长为 paddedSkv = ceil(Skv/2048)*2048（TopK 整块读取契约）
#define PADDED_SKV (((Skv + 2047) / 2048) * 2048)
#define OUT_INDICES (OUT_SCORES + (uint64_t)Sq * PADDED_SKV * 4)
// key/hist/prefix scratch 紧随 indices 之后
#define KEY_SCRATCH (OUT_INDICES + (uint64_t)Sq * topK * 4 + 8192 + (uint64_t)Sq * PADDED_SKV * 4 + 2048)

#ifndef Batch
#define Batch 1
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

static_assert(Batch == 1,
              "original dynamic QLI driver supports exactly one batch");
static_assert(Sq > 0 && Sq <= 8192 && Skv > 0 && Skv <= 16384,
              "dynamic QLI dimensions exceed the supported bounds");
static_assert(Skv % 32 == 0,
              "dynamic QLI Skv must be divisible by 32");
static_assert(topK > 0 && topK <= Skv,
              "dynamic QLI topK must be in the range 1..Skv");
static_assert(numPEs > 0 && Sq % numPEs == 0,
              "dynamic QLI requires positive NPE dividing Sq");

int main() {
    using dtype = DTYPE;
    dtype*   q       = reinterpret_cast<dtype*>(_binary_srcq_data_start);
    dtype*   k       = reinterpret_cast<dtype*>(_binary_srck_data_start);
    float*   w       = reinterpret_cast<float*>(_binary_srcw_data_start);
    float*   scale_q = reinterpret_cast<float*>(_binary_srcsq_data_start);
    float*   scale_k = reinterpret_cast<float*>(_binary_srcsk_data_start);
    float*   scores  = reinterpret_cast<float*>(OUT_SCORES);
    int32_t* indices = reinterpret_cast<int32_t*>(OUT_INDICES);
    uint32_t* key_scratch = reinterpret_cast<uint32_t*>(KEY_SCRATCH);

    // v0.58.4：W*scale_q 预广播 [Sq*g, kTk]；K 转置 [D, Skv] 行主序（CUBE_N8 契约）
    static float wbb[Sq * 64 * 32];
    for (int r = 0; r < Sq * 64; r++)
        for (int c = 0; c < 32; c++)
            wbb[r * 32 + c] = w[r] * scale_q[r];
    // K^T 按 [kTk 列块] 分块连续存放：块 j 为 [128, 32] 行主序
    static dtype ktt[128 * Skv];
    for (int j = 0; j < Skv / 32; j++)
        for (int d = 0; d < 128; d++)
            for (int c = 0; c < 32; c++)
                ktt[j * 128 * 32 + d * 32 + c] = k[(j * 32 + c) * 128 + d];
    // 多 PE 路径每 PE 写 temp_gm + tid*kTm*kTk（512 floats/PE），须按 PE 数分配
    static float tmp16[16 * 32 * numPEs];

    BENCHSTART;
    for (int i = 0; i < Batch; i++) {
        qli_pto_dynamic<dtype>(
            scores + i * Sq * PADDED_SKV,
            q + i * Sq * 64 * 128, ktt + i * 128 * Skv,
            wbb + i * Sq * 64 * 32, scale_k,
            Sq, Skv, numPEs, tmp16);
    }
    BENCHEND;

    __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);
    for (int i = 0; i < Batch; i++) {
        qli_topk_radix_dynamic(
            scores + i * Sq * PADDED_SKV,
            indices + i * Sq * topK,
            Sq, Skv, topK, numPEs,
            key_scratch + i * Sq * PADDED_SKV);
    }
    __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

    return 0;
}
