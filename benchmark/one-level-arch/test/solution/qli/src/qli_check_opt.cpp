#include <cstdint>

#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "solution/qli/qli_pto_opt_simple.hpp"
#include "multi_thread/topk/topk_tiled.hpp"

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

// TopK 阶段使用 kernels/multi_thread/topk 的 topk_tiled 引擎（4-PE SPMD）：
//   - Skv 可达 kColsMax=131072、topk <= kTopKMax=1024（本用例仍为编译期 topK）
//   - Sq 超过 kBatchMax(4) 时按批分块串行执行，块间栅栏同步（scratch 复用）
//   - fp32/fp16 精度由 TOPK_TILED_FP32_REFINE 编译期选择（Makefile TOPK_PREC）
// 运行须 gfrun -s softcore.multiThreadNum=4（1 PE 会在栅栏死锁）。
static_assert(topK <= topk_tiled::kTopKMax, "topK must fit kTopKMax(1024)");
// QLI 分数阶段要求 Skv % kTk == 0；topk_tiled 尾块 TLOAD 按 kLane 读满，
// Skv % kLane == 0 同时保证末行末块不会越出 scores 区域
static_assert(Skv % topk_tiled::kLane == 0, "Skv must be a multiple of 32");

namespace {

// round 递增的全自旋 4-PE 屏障：每轮所有 PE 互相等待，可重复使用。
struct QliBarrier {
    volatile std::uint32_t arrive[4];
};

inline void qli_barrier(QliBarrier &bar, std::uint32_t tid, std::uint32_t round) {
    bar.arrive[tid] = round;
    __asm__ volatile("" : : : "memory");
    for (std::uint32_t pe = 0; pe < 4; ++pe) {
        while (bar.arrive[pe] < round) {
        }
    }
    __asm__ volatile("" : : : "memory");
}

QliBarrier topk_barrier{};
topk_tiled::Scratch topk_scratch[topk_tiled::kBatchMax];
topk_tiled::TopkTilingData topk_tiling;
std::int32_t topk_starts[topk_tiled::kBatchMax];
std::int32_t topk_ends[topk_tiled::kBatchMax];
std::int32_t topk_errors[topk_tiled::kBatchMax];

}  // namespace

int main(){
    using dtype = DTYPE;
    const std::uint32_t tid = get_thread_idx();
    if (tid >= 4) return 0;

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
    // v0.58.4：K 转置为 [D, Skv] 行主序（CUBE_N8 B-tile 契约）
    static dtype ktt[D * Skv];
    // CUBE->Vec 桥接临时区
    static float tmp16[kTm * kTk];

    // 数据预处理 + 分数计算仅 PE0，其余 PE 在首个栅栏处等待
    if (tid == 0) {
        for (int r = 0; r < Sq * g; r++)
            for (int c = 0; c < kTk; c++)
                wbb[r * kTk + c] = w[r] * scale_q[r];
        for (int n = 0; n < Skv; n++)
            for (int d = 0; d < D; d++)
                ktt[d * Skv + n] = k[n * D + d];

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
    }

    // Step7 (TopK) 独立计时区间：topk_tiled 4-PE，Sq > kBatchMax 时按批分块。
    // PE0 每块写 tiling/starts/ends 后放行 -> 4 PE 各处理本块的一个 batch ->
    // 块末栅栏同步（scratch/starts 复用安全）。首轮栅栏同时兜住分数就绪。
    std::uint32_t round = 1;
    __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);
    for(int i=0;i<B;i++){
        float* scores_i  = reinterpret_cast<float*>(OUT_SCORES) + (uint64_t)i*Sq*Skv;
        int32_t* indices_i = reinterpret_cast<int32_t*>(OUT_INDICES) + (uint64_t)i*Sq*topK;
        for (int base = 0; base < Sq; base += topk_tiled::kBatchMax) {
            const int chunk = (Sq - base < topk_tiled::kBatchMax)
                                  ? (Sq - base) : topk_tiled::kBatchMax;
            if (tid == 0) {
                topk_tiling.batch = chunk;
                topk_tiling.cols = Skv;
                topk_tiling.topk = topK;
                for (int b = 0; b < chunk; ++b) {
                    topk_starts[b] = 0;
                    topk_ends[b] = Skv;
                }
            }
            qli_barrier(topk_barrier, tid, round++);
            topk_tiled::run(indices_i + (uint64_t)base * topK, topk_errors,
                            scores_i + (uint64_t)base * Skv, topk_starts,
                            topk_ends, topk_scratch, &topk_tiling);
            qli_barrier(topk_barrier, tid, round++);
        }
    }
    __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

    return 0;
}
