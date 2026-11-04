#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "solution/qli/qli_pto_opt_simple.hpp"

// P1 optimization: eliminate copy_bytes — pass .data segment absolute
// addresses directly to the kernel. The kernel's global_tensor + TLOAD
// can read from any mapped address, avoiding the byte-by-byte scalar copy
// that generated ~1.14M STD blocks (92.8% of total).
//
// 输入数据经 qli_check_data.s 的 .incbin 嵌入 .data 段。直接引用链接器导出的
// _binary_*_data_start 符号，取代硬编码地址 + fix_cpp_addrs.py 事后回填：
// 地址由链接器解析，与配置/数据 size 无关，任意变体免手工回填、免逐配置重编。
extern "C" unsigned char _binary_srcq_data_start[], _binary_srck_data_start[],
    _binary_srcw_data_start[], _binary_srcsq_data_start[], _binary_srcsk_data_start[];
#define SRCQ_ADDR   ((uint64_t)_binary_srcq_data_start)
#define SRCK_ADDR   ((uint64_t)_binary_srck_data_start)
#define SRCW_ADDR   ((uint64_t)_binary_srcw_data_start)
#define SRCSQ_ADDR  ((uint64_t)_binary_srcsq_data_start)
#define SRCSK_ADDR  ((uint64_t)_binary_srcsk_data_start)

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

// 多 PE 测试路径（Makefile -DTNPE=$(NPE)，默认 4）：
//   wbb 预广播（行块）与 K^T 转置（token 列块）拆分到 NPE 个 PE 并行——两者
//   均为不相交写，无需同步；屏障 1 等全部数据块就绪后，分数阶段按行区间
//   [tid*per, ...) 拆到 NPE 个 PE（qli_pto 行区间参数，各 PE 传不相交的
//   temp_gm）；屏障 2 等全部分数行就绪后，PE0 独跑 qli_topk_radix TopK
//   （其 scratch 布局随模板 Sq 推导，按行实例化会互相踩踏，行级并行待
//   kernel 侧 scratch 参数化）。
// 运行：gfrun/gfsim -s softcore.multiThreadNum=$(NPE)（gfsim 对含 PEID 指令
// 的 ELF 自动套用 fourpe bundle，NPE<4 时多余 PE 在 tid 门控处退出）。
#ifndef TNPE
#define NPE 4
#else
#define NPE TNPE
#endif
static_assert(NPE >= 2 && NPE <= 4, "NPE must be 2..4 (multi-PE only)");

// 阶段完成标志（.bss 零初始化；volatile + 编译器屏障，与
// multi_thread_res_check.h 的发布/等待语义同款）
static volatile uint32_t prep_ready[NPE];
static volatile uint32_t scores_ready[NPE];

int main(){
    using dtype = DTYPE;

    const uint32_t tid = get_thread_idx();
    if (tid >= NPE) return 0;

    // P1: No copy_bytes — pass .data segment addresses directly.
    // The kernel's global_tensor accepts raw pointers; TLOAD reads from
    // the mapped .data region without needing aligned stack buffers.
    dtype*   q       = reinterpret_cast<dtype*>(SRCQ_ADDR);
    dtype*   k       = reinterpret_cast<dtype*>(SRCK_ADDR);
    float*   w       = reinterpret_cast<float*>(SRCW_ADDR);
    float*   scale_q = reinterpret_cast<float*>(SRCSQ_ADDR);
    float*   scale_k = reinterpret_cast<float*>(SRCSK_ADDR);

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
    // CUBE->Vec 桥接临时区：每 PE 一份（分数阶段并发调用 qli_pto）
    static float tmp16[kTm * kTk * NPE];

    // ---- 屏障 1：等全部 wbb/ktt 块就绪（每 PE 的分数行需要完整 K^T）----
    __asm__ volatile("" : : : "memory");
    prep_ready[tid] = 1;
    for (uint32_t p = 0; p < NPE; ++p) {
        while (prep_ready[p] == 0) {
        }
    }
    __asm__ volatile("" : : : "memory");

    // ---- 分数阶段：行区间 [tid*per, ...) 拆到 NPE 个 PE（A2）----
    const int per = (Sq + NPE - 1) / NPE;
    const int row0 = (int)tid * per;
    const int row1 = (row0 + per < Sq) ? (row0 + per) : Sq;
    BENCHSTART;
    for(int i=0;i<B;i++){
        qli_pto<dtype, Sq, Skv, D, g, kTm, kTk>(
            reinterpret_cast<float*>(OUT_SCORES) + i*Sq*Skv,
            q + i*Sq*g*D,
            ktt + i*D*Skv,
            wbb + i*Sq*g*kTk,
            scale_k + i*Skv,
            tmp16 + (size_t)tid * kTm * kTk,
            row0, row1
        );
    }
    BENCHEND;

    // ---- 屏障 2：等全部分数行就绪后，PE0 独跑 TopK ----
    __asm__ volatile("" : : : "memory");
    scores_ready[tid] = 1;
    if (tid != 0) return 0;
    for (uint32_t p = 1; p < NPE; ++p) {
        while (scores_ready[p] == 0) {
        }
    }
    __asm__ volatile("" : : : "memory");

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
