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
//   - 动态 shape：gM/gN/gK 运行时传入（内核模板只有 tM/tN/tK，
//     要求 M/N/K 被 tile 整除）
//   - 运行需 4 线程: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
//   - 精度: res_check=on 构建后由 verify_matmul_test.py（--multi-thread）比对；
//     非 res_check 构建用确定性 LCG hard 随机 fp16（host 侧可位级复现）

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

// 确定性 LCG（与单线程 matmul_test.cpp 相同算法，host 侧可位级复现）。
static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

// hard 随机 fp16 位模式填充（sign 随机、exp ∈ [8,16)、mantissa 全随机），
// 直接按位写入，不经浮点转换。
static void fill_hard_half(__half *buf, int rows, int cols, uint64_t seed) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const uint64_t rv = lcg_next(&seed);
            const uint16_t sign = (uint16_t)(rv & 1ull);
            const uint16_t exp = (uint16_t)(8ull + ((rv >> 1) & 7ull));
            const uint16_t man = (uint16_t)((rv >> 4) & 0x3FFull);
            const uint16_t bits =
                (uint16_t)((sign << 15) | (exp << 10) | man);
            memcpy(&buf[(size_t)r * cols + c], &bits, sizeof(bits));
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
    for (int b = 0; b < Batch; ++b) {
        matmul_test_mt<tilM, tilN, tilK>(
            context->dst + b * globM * globN,
            context->src0 + b * globM * globK,
            context->src1 + b * globK * globN,
            context->scratch, globM, globN, globK);
    }
    BENCHEND;
    return 0;
}

int main() {
    constexpr int kPeNum = 4;
    const uint32_t tid = get_thread_idx();

    static_assert(globM % tilM == 0, "global M must be divisible by tM");
    static_assert(globN % tilN == 0, "global N must be divisible by tN");
    static_assert(globK % tilK == 0, "global K must be divisible by tK");

    static __half src0p[Batch * globM * globK + 2 * ALIGN];
    static __half src1p[Batch * globK * globN + 2 * ALIGN];
    static __half dstp[Batch * globM * globN + 2 * ALIGN];
    // fp32 scratch：每 PE 一个 [tM/4, tN] 切片，共 4 片。
    static float scratchp[kPeNum * (tilM / kPeNum) * tilN + 2 * ALIGN];

    __half *src0 = (__half *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    __half *src1 = (__half *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    __half *dst = (__half *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);
    float *scratch = (float *)(((uint64_t)scratchp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == 0) {
        readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                       Batch * globM * globK * sizeof(__half));
        readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                       Batch * globK * globN * sizeof(__half));
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#else
    // 功能跑通验证：hard 随机 fp16。SPMD（无 group runtime）时 4 线程各自
    // 填充同一份静态数据，值相同幂等；group runtime 时 main 仅 PE0 执行。
    fill_hard_half(src0, Batch * globM, globK, 0x123456789ABCDEF0ull);
    fill_hard_half(src1, Batch * globK, globN, 0x0FEDCBA987654321ull);
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
        writeBinaryFile(RES_PATH, (uint8_t *)dst,
                        Batch * globM * globN * sizeof(__half));
    }
#endif

    return status;
}
