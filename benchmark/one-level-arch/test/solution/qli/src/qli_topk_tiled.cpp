#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "solution/qli/qli_pto_opt_simple.hpp"
#include "multi_thread/topk/topk_tiled.hpp"

// qli_topk_tiled — QLI 端到端（分数阶段 + tiled TopK 阶段），4-PE hosted res_check。
//
// 把 QLI 的 TopK 阶段替换为 kernels/multi_thread/topk 的 topk_tiled 引擎
//（256-bin FP16-sortable-key 直方图 radix-select + 边界桶精化）：
//   - Skv 可达 topk_tiled::kColsMax = 131072（DeepSeek-v4 风格 128K->1024）
//   - topk 为运行时参数（shape.bin，<= topk_tiled::kTopKMax = 1024）
//   - fp32 / fp16 精度为编译期开关（-DTOPK_TILED_FP32_REFINE=1/0，默认 fp32）
//   - 批（Sq）<= topk_tiled::kBatchMax = 4，topk_tiled 按 bx = tid..batch 步进 4 分行
//
// 数据流（CHK_DIR 由 Makefile res_check=on 注入，文件 I/O 仅 PE0）：
//   读入: shape.bin {batch,cols,topk} / srcq.bin [Sq*g,D] FP8
//         srckt.bin [D,Skv] FP8（host 预转置的 K^T，免设备侧 16M 标量转置）
//         srcw.bin [Sq*g] FP32 / srcsq.bin [Sq*g] FP32 / srcsk.bin [Skv] FP32
//   写出: output.bin [Sq,topk] I32 / errors.bin [Sq] I32
//         scores_readback.bin [Sq,Skv] FP32（分数链路核对）
//
// PE 协作（三栅栏，round 递增的全自旋屏障）：
//   barrier(1): PE0 输入读入完成后发布，其余 PE 等待
//   barrier(2): PE0 完成 qli_pto 分数计算（单 PE）后放行，4 PE 进入 TopK
//   barrier(3): 4 PE 完成 TopK（各写各行 batch 的不相交输出）后 PE0 导出
//
// 运行: gfrun -s softcore.multiThreadNum=4 -f <elf>（4-PE hosted，1 PE 会栅栏死锁）

#ifndef Tsq
#define Sq 1
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 131072
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

#ifndef B
#define B 1
#endif

#define DTYPE __fp8_e4m3

// topk_tiled 的批上限即 QLI 行数上限（topk_tiled 按 batch 划分 4 PE 工作）
static_assert(Sq <= topk_tiled::kBatchMax, "qli_topk_tiled needs Sq <= kBatchMax(4)");
static_assert(Skv <= topk_tiled::kColsMax, "qli_topk_tiled needs Skv <= kColsMax(131072)");

#define OUT_SCORES 0x4000802000ULL
// scores 尾部留 kLane 个 float 余量（topk_tiled 尾块 TLOAD 会读满 32 lane）
#define OUT_INDICES (0x4000802000ULL + (uint64_t)((Sq * Skv + topk_tiled::kLane) * 4))

namespace {

// round 递增的全自旋 4-PE 屏障：每个 round 所有 PE 互相等待（可复用，
// MultiThreadResCheckSync 的 done[] 语义只支持一轮）。
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

struct Buffers {
    alignas(4096) DTYPE srcq[Sq * g * D];      // Q [Sq*g, D] FP8
    alignas(4096) DTYPE srckt[D * Skv];        // K^T [D, Skv] FP8（host 预转置）
    alignas(4096) float srcw[Sq * g];          // W [Sq*g]
    alignas(4096) float srcsq[Sq * g];         // scale_q [Sq*g]
    alignas(4096) float srcsk[Skv];            // scale_k [Skv]
    alignas(4096) float wbb[Sq * g * kTk];     // W*scale_q 预广播 [Sq*g, kTk]
    alignas(4096) float tmp16[kTm * kTk];      // CUBE->Vec 桥接临时区
    alignas(4096) topk_tiled::Scratch scratch[Sq];
    alignas(4096) topk_tiled::TopkTilingData tiling;
    alignas(4096) std::int32_t starts[Sq];
    alignas(4096) std::int32_t ends[Sq];
    alignas(4096) std::int32_t errors[Sq];
};

Buffers buffers{};
QliBarrier barrier_state{};

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    if (tid >= 4) return 0;

    if (tid == 0) {
        readBinaryFile(CHK_DIR "/shape.bin",
                       reinterpret_cast<std::uint8_t *>(&buffers.tiling),
                       sizeof(buffers.tiling));
        readBinaryFile(CHK_DIR "/srcq.bin",
                       reinterpret_cast<std::uint8_t *>(buffers.srcq),
                       sizeof(buffers.srcq));
        readBinaryFile(CHK_DIR "/srckt.bin",
                       reinterpret_cast<std::uint8_t *>(buffers.srckt),
                       sizeof(buffers.srckt));
        readBinaryFile(CHK_DIR "/srcw.bin",
                       reinterpret_cast<std::uint8_t *>(buffers.srcw),
                       sizeof(buffers.srcw));
        readBinaryFile(CHK_DIR "/srcsq.bin",
                       reinterpret_cast<std::uint8_t *>(buffers.srcsq),
                       sizeof(buffers.srcsq));
        readBinaryFile(CHK_DIR "/srcsk.bin",
                       reinterpret_cast<std::uint8_t *>(buffers.srcsk),
                       sizeof(buffers.srcsk));
    }
    qli_barrier(barrier_state, tid, 1);

    if (tid == 0) {
        // W*scale_q 预广播（行 r 全列同值）——与 qli_check_opt 相同
        for (int r = 0; r < Sq * g; ++r)
            for (int c = 0; c < kTk; ++c)
                buffers.wbb[r * kTk + c] = buffers.srcw[r] * buffers.srcsq[r];

        float *scores = reinterpret_cast<float *>(OUT_SCORES);
        DTYPE *q = buffers.srcq;
        DTYPE *k = buffers.srckt;  // 已是 K^T [D, Skv] 布局，免设备侧转置
        for (int i = 0; i < B; ++i) {
            qli_pto<DTYPE, Sq, Skv, D, g, kTm, kTk>(
                scores + (std::uint64_t)i * Sq * Skv,
                q + (std::uint64_t)i * Sq * g * D,
                k + (std::uint64_t)i * D * Skv,
                buffers.wbb + (std::uint64_t)i * Sq * g * kTk,
                buffers.srcsk + (std::uint64_t)i * Skv,
                buffers.tmp16);
        }
    }
    qli_barrier(barrier_state, tid, 2);

    // TopK：topk_tiled 按 batch（=QLI 行）在 4 PE 间分行
    for (int b = 0; b < Sq; ++b) {
        buffers.starts[b] = 0;
        buffers.ends[b] = Skv;
        buffers.errors[b] = 0;
    }
    topk_tiled::run(reinterpret_cast<std::int32_t *>(OUT_INDICES), buffers.errors,
                    reinterpret_cast<const float *>(OUT_SCORES), buffers.starts,
                    buffers.ends, buffers.scratch, &buffers.tiling);
    qli_barrier(barrier_state, tid, 3);

    if (tid == 0) {
        const std::int32_t batch = static_cast<std::int32_t>(buffers.tiling.batch);
        const std::int32_t topk = static_cast<std::int32_t>(buffers.tiling.topk);
        const std::int32_t *out = reinterpret_cast<const std::int32_t *>(OUT_INDICES);
        const float *scores = reinterpret_cast<const float *>(OUT_SCORES);
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<const std::uint8_t *>(out),
                        (std::size_t)batch * topk * sizeof(std::int32_t));
        writeBinaryFile(CHK_DIR "/errors.bin",
                        reinterpret_cast<const std::uint8_t *>(buffers.errors),
                        (std::size_t)Sq * sizeof(std::int32_t));
        writeBinaryFile(CHK_DIR "/scores_readback.bin",
                        reinterpret_cast<const std::uint8_t *>(scores),
                        (std::size_t)Sq * Skv * sizeof(float));
    }
    return 0;
}
