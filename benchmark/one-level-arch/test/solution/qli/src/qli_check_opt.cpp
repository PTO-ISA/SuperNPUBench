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

// 多 PE 测试路径（Makefile -DTNPE=$(NPE)，默认 4）：
//   wbb 预广播（行块）与 K^T 转置（token 列块）拆分到 NPE 个 PE 并行——两者
//   均为不相交写，无需同步；worker 完成各自块后置完成标志即退出，仅 PE0
//   自旋等待全部 ktt 就绪，随后独跑 qli_pto 分数阶段与 qli_topk_radix TopK
//   （单 PE 代码路径不变）。分数/TopK 的行级并行需 kernel 侧改造
//   （qli_topk_radix 的 scratch 布局随模板 Sq 推导，按行实例化会互相踩踏）。
// 运行：gfrun/gfsim -s softcore.multiThreadNum=$(NPE)（gfsim 对含 PEID 指令
// 的 ELF 自动套用 fourpe bundle，NPE<4 时多余 PE 在 tid 门控处退出）。
#ifndef TNPE
#define NPE 4
#else
#define NPE TNPE
#endif
static_assert(NPE >= 1 && NPE <= 4, "NPE must be in the range 1..4");

// 完成标志（.bss 零初始化；volatile + 编译器屏障，与
// multi_thread_res_check.h 的发布/等待语义同款）
static volatile uint32_t ktt_ready[NPE];

// Only PE0 calls this boundary after the shared preprocessing completes.
// Its static Tile domain covers assembly writers, views, element operations,
// and compiler-generated copies in every reachable helper.
PTO_PE_DOMAIN(8)
static void run_qli_pe0(DTYPE *q, DTYPE *ktt, float *wbb,
                       float *scale_k, float *tmp16) {
    using dtype = DTYPE;
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

}

int main(){
    using dtype = DTYPE;

    const uint32_t tid = get_thread_idx();
    if (tid >= NPE) return 0;

    dtype*   q       = reinterpret_cast<dtype*>(_binary_srcq_data_start);
    dtype*   k       = reinterpret_cast<dtype*>(_binary_srck_data_start);
    float*   w       = reinterpret_cast<float*>(_binary_srcw_data_start);
    float*   scale_q = reinterpret_cast<float*>(_binary_srcsq_data_start);
    float*   scale_k = reinterpret_cast<float*>(_binary_srcsk_data_start);

    // v0.58.4：W*scale_q 预广播为 [Sq*g, kTk]（行 r 全列同值），
    // kernel 内用普通 TMUL（规避单列广播源的物理列校验）；
    // 多 PE：行区间 [tid*per, ...) 拆分（不相交写）
    static float wbb[Sq * g * kTk];
    {
        const int total = Sq * g;
        const int per = (total + NPE - 1) / NPE;
        const int r0 = (int)tid * per;
        const int r1 = (r0 + per < total) ? (r0 + per) : total;
        for (int r = r0; r < r1; r++) {
            const float v = w[r] * scale_q[r];
            // unroll(disable)：gfsim fourpe bundle 的 BCC 提交协议对"双 load +
            // fmul + 多 store 展开"的 [FP COND] 循环块会确定性卡死（cycles=13939，
            // blocks=59，27 块全堵 STID0）；禁用内层展开后正常推进。gfrun 不受影响。
#pragma clang loop unroll(disable)
            for (int c = 0; c < kTk; c++)
                wbb[r * kTk + c] = v;
        }
    }
    // v0.58.4：K 转置为 [D, Skv] 行主序（CUBE_N8 B-tile 契约）；
    // 多 PE：token 列区间 [tid*per, ...) 拆分（不相交写）
    static dtype ktt[D * Skv];
    {
        const int per = (Skv + NPE - 1) / NPE;
        const int n0 = (int)tid * per;
        const int n1 = (n0 + per < Skv) ? (n0 + per) : Skv;
        for (int n = n0; n < n1; n++)
            for (int d = 0; d < D; d++)
                ktt[d * Skv + n] = k[n * D + d];
    }
    // CUBE->Vec 桥接临时区
    static float tmp16[kTm * kTk];

    // worker：各自块完成 → 置标志 → 退出；PE0：等全部 ktt 块就绪
    __asm__ volatile("" : : : "memory");
    ktt_ready[tid] = 1;
    if (tid != 0) return 0;
    for (uint32_t p = 1; p < NPE; ++p) {
        while (ktt_ready[p] == 0) {
        }
    }
    __asm__ volatile("" : : : "memory");

    run_qli_pe0(q, ktt, wbb, scale_k, tmp16);
    return 0;
}
