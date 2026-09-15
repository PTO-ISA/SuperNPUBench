#include "solution/quant_batch_matmul/quant_batch_matmul_mxfp4_mt.hpp"

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

// quant_batch_matmul_test_mxfp4_mt — 4-PE cooperative e2m1x2 fp4 MX matmul 测试入口
//
// 4 个 PE 协作执行 TMATMUL_MX：A/B 通过 SharedTile 装入共享瓦片，
// 每 PE 持有并写回 [kPeM=tM/4, tN] 行切片。A/B/scale 均为全局共享矩阵。
//
// A/B 为 e2m1x2 fp4 打包（__fp4_e2m1x2，每字节 2 个元素），
// MX scaling 用 e8m0 / 32-element group。
//
// 运行: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
// 精度: res_check=on + verify_quant_batch_matmul_test.py（--multi-thread）

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
#define tilM 64
#endif

#ifndef tilN
#define tilN 32
#endif

#ifndef tilK
#define tilK 64
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

// 确定性 LCG（host 侧可位级复现）。
static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

// hard 随机按字节填充 fp4 打包数据与 e8m0 scale。
static void fill_hard_bytes(uint8_t *buf, size_t numel, uint64_t seed) {
    for (size_t i = 0; i < numel; ++i) {
        buf[i] = (uint8_t)(lcg_next(&seed) & 0xFFull);
    }
}

struct QuantBatchMatmulMxfp4MtContext {
    float *dst;
    __fp4_e2m1x2 *src0;
    __fp4_e2m1x2 *src1;
    __fp8_e8m0 *src0_mx;
    __fp8_e8m0 *src1_mx;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    QuantBatchMatmulMxfp4MtContext *context =
        static_cast<QuantBatchMatmulMxfp4MtContext *>(opaque);

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        constexpr int globKv = globK / 2;
        constexpr int tilKv = tilK / 2;
        size_t bOffA  = (size_t)b * globM * globKv;
        size_t bOffB  = (size_t)b * globKv * globN;
        size_t bOffC  = (size_t)b * globM * globN;
        size_t bOffAS = (size_t)b * globM * (globKv / 32);
        size_t bOffBS = (size_t)b * (globKv / 32) * globN;

        quant_batch_matmul_mxfp4_mt<tilM, tilN, tilKv, globM, globN, globKv>(
            context->dst + bOffC,
            context->src0 + bOffA,
            context->src1 + bOffB,
            context->src0_mx + bOffAS,
            context->src1_mx + bOffBS);
    }
    BENCHEND;
    return 0;
}

int main() {
    const uint32_t tid = get_thread_idx();

    constexpr int globKv = globK / 2;
    constexpr int tilKv = tilK / 2;

    static_assert(tilM == 64 || tilM == 128,
                  "4-PE cooperative requires tilM == 64 or 128");
    static_assert(globM % tilM == 0, "globM must be divisible by tilM");
    static_assert(globN % tilN == 0, "globN must be divisible by tilN");
    static_assert(globKv % tilKv == 0, "globKv must be divisible by tilKv");
    static_assert(globK % 32 == 0, "globK must be divisible by 32 (MX scale group)");

    static __fp4_e2m1x2 src0p[Batch * globM * globKv + 2 * ALIGN];
    static __fp4_e2m1x2 src1p[Batch * globKv * globN + 2 * ALIGN];
    static __fp8_e8m0 src0_mxp[Batch * globM * (globK / 32) + 2 * ALIGN];
    static __fp8_e8m0 src1_mxp[Batch * (globK / 32) * globN + 2 * ALIGN];
    static float dstp[Batch * globM * globN + 2 * ALIGN];

    __fp4_e2m1x2 *src0 = (__fp4_e2m1x2 *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    __fp4_e2m1x2 *src1 = (__fp4_e2m1x2 *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    __fp8_e8m0 *src0_mx = (__fp8_e8m0 *)(((uint64_t)src0_mxp & ALIGN_MASK) + ALIGN);
    __fp8_e8m0 *src1_mx = (__fp8_e8m0 *)(((uint64_t)src1_mxp & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH    CHK_DIR "/src0.bin"
#define SRC1_PATH    CHK_DIR "/src1.bin"
#define SRC0_MX_PATH CHK_DIR "/src0_mx.bin"
#define SRC1_MX_PATH CHK_DIR "/src1_mx.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == 0) {
        readBinaryFile(SRC0_PATH,    (uint8_t *)src0,    Batch * globM * globKv * sizeof(__fp4_e2m1x2));
        readBinaryFile(SRC1_PATH,    (uint8_t *)src1,    Batch * globKv * globN * sizeof(__fp4_e2m1x2));
        readBinaryFile(SRC0_MX_PATH, (uint8_t *)src0_mx, Batch * globM * (globK / 32) * sizeof(__fp8_e8m0));
        readBinaryFile(SRC1_MX_PATH, (uint8_t *)src1_mx, Batch * (globK / 32) * globN * sizeof(__fp8_e8m0));
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#else
    // 功能跑通验证：hard 随机 fp4/e8m0 位模式填充。
    fill_hard_bytes((uint8_t *)src0, Batch * globM * globKv, 0x123456789ABCDEF0ull);
    fill_hard_bytes((uint8_t *)src1, Batch * globKv * globN, 0x0FEDCBA987654321ull);
    fill_hard_bytes((uint8_t *)src0_mx, Batch * globM * (globK / 32), 0xAABBCCDDEEFF0011ull);
    fill_hard_bytes((uint8_t *)src1_mx, Batch * (globK / 32) * globN, 0x2233445566778899ull);
#endif

    QuantBatchMatmulMxfp4MtContext context{dst, src0, src1, src0_mx, src1_mx};
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
        writeBinaryFile(RES_PATH, (uint8_t *)dst, Batch * globM * globN * sizeof(float));
    }
#endif

    return status;
}
