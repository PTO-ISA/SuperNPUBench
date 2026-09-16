#include "solution/quant_batch_matmul/quant_batch_matmul_hif4.hpp"

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

// quant_batch_matmul_test_hif4 — 4-PE cooperative HiF4X2 matmul 测试入口
//
// A/B 为 hif4x2 打包（__fp4_hif4x2，每字节 2 个元素），
// MX scaling 用 uint32_t U32 scale / 64-element group (logical)。
// B 存储为 [N, K_stored] row-major，B scale 为 [N, K/64]。
//
// 运行: gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>

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

static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

static void fill_hard_bytes(uint8_t *buf, size_t numel, uint64_t seed) {
    for (size_t i = 0; i < numel; ++i) {
        buf[i] = (uint8_t)(lcg_next(&seed) & 0xFFull);
    }
}

struct QuantBatchMatmulHif4MtContext {
    float *dst;
    __fp4_hif4x2 *src0;
    __fp4_hif4x2 *src1;
    uint32_t *src0_mx;
    uint32_t *src1_mx;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    QuantBatchMatmulHif4MtContext *context =
        static_cast<QuantBatchMatmulHif4MtContext *>(opaque);

    constexpr int kPackedFactor = 2;
    constexpr int kStoredGK = globK / kPackedFactor;
    constexpr int kGScaleK = globK / 64;  // 64 logical per U32

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        size_t bOffA  = (size_t)b * globM * kStoredGK;
        size_t bOffB  = (size_t)b * globN * kStoredGK;
        size_t bOffC  = (size_t)b * globM * globN;
        size_t bOffAS = (size_t)b * globM * kGScaleK;
        size_t bOffBS = (size_t)b * globN * kGScaleK;

        quant_batch_matmul_hif4_mt<tilM, tilN, tilK, globM, globN, globK>(
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

    constexpr int kPackedFactor = 2;
    constexpr int kStoredGK = globK / kPackedFactor;
    constexpr int kGScaleK = globK / 64;

    static_assert(tilM == 64 || tilM == 128,
                  "4-PE cooperative requires tilM == 64 or 128");
    static_assert(globM % tilM == 0, "globM must be divisible by tilM");
    static_assert(globN % tilN == 0, "globN must be divisible by tilN");
    static_assert(globK % tilK == 0, "globK must be divisible by tilK");
    static_assert(globK % 64 == 0, "globK must be divisible by 64 (HiF4 scale group)");

    static __fp4_hif4x2 src0p[Batch * globM * kStoredGK + 2 * ALIGN];
    static __fp4_hif4x2 src1p[Batch * globN * kStoredGK + 2 * ALIGN];
    static uint32_t src0_mxp[Batch * globM * kGScaleK + 2 * ALIGN];
    static uint32_t src1_mxp[Batch * globN * kGScaleK + 2 * ALIGN];
    static float dstp[Batch * globM * globN + 2 * ALIGN];

    __fp4_hif4x2 *src0 = (__fp4_hif4x2 *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    __fp4_hif4x2 *src1 = (__fp4_hif4x2 *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    uint32_t *src0_mx = (uint32_t *)(((uint64_t)src0_mxp & ALIGN_MASK) + ALIGN);
    uint32_t *src1_mx = (uint32_t *)(((uint64_t)src1_mxp & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH    CHK_DIR "/src0.bin"
#define SRC1_PATH    CHK_DIR "/src1.bin"
#define SRC0_MX_PATH CHK_DIR "/src0_mx.bin"
#define SRC1_MX_PATH CHK_DIR "/src1_mx.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == 0) {
        readBinaryFile(SRC0_PATH,    (uint8_t *)src0,    Batch * globM * kStoredGK * sizeof(__fp4_hif4x2));
        readBinaryFile(SRC1_PATH,    (uint8_t *)src1,    Batch * globN * kStoredGK * sizeof(__fp4_hif4x2));
        readBinaryFile(SRC0_MX_PATH, (uint8_t *)src0_mx, Batch * globM * kGScaleK * sizeof(uint32_t));
        readBinaryFile(SRC1_MX_PATH, (uint8_t *)src1_mx, Batch * globN * kGScaleK * sizeof(uint32_t));
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#else
    fill_hard_bytes((uint8_t *)src0, Batch * globM * kStoredGK, 0x123456789ABCDEF0ull);
    fill_hard_bytes((uint8_t *)src1, Batch * globN * kStoredGK, 0x0FEDCBA987654321ull);
    fill_hard_bytes((uint8_t *)src0_mx, Batch * globM * kGScaleK * sizeof(uint32_t), 0xAABBCCDDEEFF0011ull);
    fill_hard_bytes((uint8_t *)src1_mx, Batch * globN * kGScaleK * sizeof(uint32_t), 0x2233445566778899ull);
#endif

    QuantBatchMatmulHif4MtContext context{dst, src0, src1, src0_mx, src1_mx};
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
