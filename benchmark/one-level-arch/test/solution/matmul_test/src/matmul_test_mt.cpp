#include "solution/matmul_test/matmul_test_mt.hpp"

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

// matmul_test_mt 测试入口（4-PE 多线程动态 shape fp16 GEMM）:
//   - 输入 fp16 (__half)，累加 fp32，输出 fp16 (__half)
//   - 动态 shape：gM/gN/gK 运行时传入；内核对 M/N 做零填充
//     （Mpad/Npad = ceil(dim/tile)*tile），K 要求被 tK 整除
//   - 缓冲布局：A [Mpad, gK]（行距 gK）、B [gK, Npad]（行距 Npad）、
//     C [Mpad, Npad]（行距 Npad）；逻辑结果 = C[0:gM, 0:gN]
//   - 运行需 4 线程: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
//   - 精度: 非 res_check 构建用确定性 LCG hard 随机 fp16（host 侧可位级
//     复现），--dump-memory + Test/verify_matmul_test_mt_direct.py 比对

#ifndef globM
#define globM 256
#endif

#ifndef globN
#define globN 256
#endif

#ifndef globK
#define globK 256
#endif

#ifndef tilM
#define tilM 128
#endif

#ifndef tilN
#define tilN 64
#endif

#ifndef tilK
#define tilK 64
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

// 填充后维度与每 PE 行数（与内核同款公式；kPeM 为契约 RowsPerPE）。
constexpr int kMpad = ((globM + tilM - 1) / tilM) * tilM;
constexpr int kNpad = ((globN + tilN - 1) / tilN) * tilN;
constexpr int kPeRows = tilM <= 64 ? 16 : 32;

// 确定性 LCG（与单线程 matmul_test.cpp 相同算法，host 侧可位级复现）。
static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

// hard 随机 fp16 位模式填充（sign 随机、exp ∈ [8,16)、mantissa 全随机），
// 直接按位写入，不经浮点转换。带 leading dimension：只填逻辑
// [rows_logic, cols_logic] 区，行距 ld；其余（填充区）保持静态缓冲的 0。
static void fill_hard_half(__half *buf, int ld, int rows_logic,
                           int cols_logic, uint64_t seed) {
    for (int r = 0; r < rows_logic; ++r) {
        for (int c = 0; c < cols_logic; ++c) {
            const uint64_t rv = lcg_next(&seed);
            const uint16_t sign = (uint16_t)(rv & 1ull);
            const uint16_t exp = (uint16_t)(8ull + ((rv >> 1) & 7ull));
            const uint16_t man = (uint16_t)((rv >> 4) & 0x3FFull);
            const uint16_t bits =
                (uint16_t)((sign << 15) | (exp << 10) | man);
            memcpy(&buf[(size_t)r * ld + c], &bits, sizeof(bits));
        }
    }
}

struct MatmulTestContext {
    __half *dst;
    __half *src0;
    __half *src1;
    float *scratch;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    MatmulTestContext *context = static_cast<MatmulTestContext *>(opaque);

    BENCHSTART;
    // 传填充后维度（constexpr 常量折叠）：内核按 [kMpad, globK] x
    // [globK, kNpad] 的精确整除问题计算。
    for (int b = 0; b < Batch; ++b) {
        matmul_test_mt<tilM, tilN, tilK>(
            context->dst + (size_t)b * kMpad * kNpad,
            context->src0 + (size_t)b * kMpad * globK,
            context->src1 + (size_t)b * globK * kNpad,
            context->scratch, kMpad, kNpad, globK);
    }
    BENCHEND;
    return 0;
}

int main() {
    constexpr int kPeNum = 4;
    const uint32_t tid = get_thread_idx();

    static_assert(globK % tilK == 0, "global K must be divisible by tK");
    static_assert(Batch == 1 || kMpad == globM,
                  "Batch > 1 requires gM to be divisible by tM (the flat "
                  "input fill assumes contiguous per-batch layouts)");
    static_assert((tilM & (tilM - 1)) == 0 && tilM >= 1 && tilM <= 128,
                  "tM must be a power of two in 1..128 (model constraint)");
    static_assert((tilN & (tilN - 1)) == 0,
                  "tN must be a power of two (model constraint)");
    static_assert((tilK & (tilK - 1)) == 0,
                  "tK must be a power of two (model constraint)");

    static __half src0p[Batch * kMpad * globK + 2 * ALIGN];
    static __half src1p[Batch * globK * kNpad + 2 * ALIGN];
    static __half dstp[Batch * kMpad * kNpad + 2 * ALIGN];
    // fp32 scratch：每 PE 一个 [kPeRows, tN] 切片，共 4 片。
    static float scratchp[kPeNum * kPeRows * tilN + 2 * ALIGN];

    __half *src0 = (__half *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    __half *src1 = (__half *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    __half *dst = (__half *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);
    float *scratch = (float *)(((uint64_t)scratchp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == 0) {
        // 逻辑 [gM,gK]/[gK,gN] 展开到带 leading dimension 的填充缓冲。
        {
            static __half sbuf[globM * globK];
            readBinaryFile(SRC0_PATH, (uint8_t *)sbuf,
                           Batch * globM * globK * sizeof(__half));
            for (int r = 0; r < Batch * globM; ++r)
                for (int c = 0; c < globK; ++c)
                    src0[(size_t)r * globK + c] = sbuf[(size_t)r * globK + c];
        }
        {
            static __half sbuf[globK * globN];
            readBinaryFile(SRC1_PATH, (uint8_t *)sbuf,
                           Batch * globK * globN * sizeof(__half));
            for (int r = 0; r < Batch * globK; ++r)
                for (int c = 0; c < globN; ++c)
                    src1[(size_t)r * kNpad + c] = sbuf[(size_t)r * globN + c];
        }
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#else
    // 功能跑通验证：hard 随机 fp16。SPMD（无 group runtime）时 4 线程各自
    // 填充同一份静态数据，值相同幂等；group runtime 时 main 仅 PE0 执行。
#if PERF_NO_FILL
    // 性能仿真专用构建：跳过 LCG 填充（gfsim 计时与数据无关）。
#else
    fill_hard_half(src0, globK, Batch * globM, globK, 0x123456789ABCDEF0ull);
    fill_hard_half(src1, kNpad, Batch * globK, globN, 0x0FEDCBA987654321ull);
#endif
#endif

    MatmulTestContext context{dst, src0, src1, scratch};
#ifdef LINX_GROUP_RUNTIME
    const int status = linx_group_run(&context);
#else
    const int status = __linx_group_worker_main(0, &context);
#endif

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
#ifndef LINX_GROUP_RUNTIME
    res_check_wait_for_all(res_check_sync, tid);
#endif
    if (tid == 0) {
        // 输出只写逻辑 gM x gN（丢弃填充区）。
        static __half resbuf[globM * globN];
        for (int r = 0; r < Batch * globM; ++r)
            for (int c = 0; c < globN; ++c)
                resbuf[(size_t)r * globN + c] = dst[(size_t)r * kNpad + c];
        writeBinaryFile(RES_PATH, (uint8_t *)resbuf,
                        Batch * globM * globN * sizeof(__half));
    }
#endif

    return status;
}
