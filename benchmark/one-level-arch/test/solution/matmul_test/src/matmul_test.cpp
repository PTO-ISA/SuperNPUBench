#include "solution/matmul_test/matmul_test.hpp"

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

// matmul_test — 4-PE cooperative fp16 GEMM 测试入口（A Local / B Shared）:
//   gfrun -t 1 -f <elf>（模型默认 4 PE）
//   精度: --dump-memory + Test/verify_matmul_test_mt_direct.py

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

// M 尾组：tilM=128 且尾部 ∈ {16,32,64}（pow2 group_M，per-PE shard 16）时，
// 用独立尾组替代整组 padding；NO_TAIL_GROUP 强制回到传统 padding 路径。
// 尾组 4 个 shard 覆盖 4*16=64 行（零行 PE 读 A 零填充行、写 C 填充行）。
#ifdef NO_TAIL_GROUP
constexpr bool kUseTail = false;
#else
constexpr bool kUseTail = (tilM == 128) && (globM % tilM != 0) &&
    ((globM % tilM == 16) || (globM % tilM == 32) || (globM % tilM == 64));
#endif
constexpr int kTailM = kUseTail ? (globM % tilM) : 0;
constexpr int kMpad = kUseTail ? (globM / tilM) * tilM + 4 * 16
                               : ((globM + tilM - 1) / tilM) * tilM;
constexpr int kNpad = ((globN + tilN - 1) / tilN) * tilN;
constexpr int kPeRows = tilM <= 64 ? 16 : 32;

// 复用模式与驻留累加器个数（可选宏，默认 0=不复用 / 4=上限）。
#ifndef REUSE_MODE
#define REUSE_MODE 0
#endif
#ifndef REUSE_COUNT
#define REUSE_COUNT 4
#endif
static_assert(REUSE_MODE >= 0 && REUSE_MODE <= 2,
              "REUSE_MODE must be 0 (none) / 1 (reuseA) / 2 (reuseB)");
static_assert(REUSE_COUNT >= 1 && REUSE_COUNT <= 4,
              "REUSE_COUNT must be in 1..4");

static inline uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}

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
    for (int b = 0; b < Batch; ++b) {
        matmul_test<tilM, tilN, tilK, kTailM, REUSE_MODE, REUSE_COUNT,
                    kMpad, kNpad, globK>(
            context->dst + (size_t)b * kMpad * kNpad,
            context->src0 + (size_t)b * kMpad * globK,
            context->src1 + (size_t)b * globK * kNpad,
            context->scratch);
    }
    BENCHEND;
    return 0;
}

int main() {
    constexpr int kPeNum = 4;
    const uint32_t tid = get_thread_idx();

    static_assert(globK % tilK == 0, "K must be divisible by tK");
    static_assert(Batch == 1 || kMpad == globM,
                  "Batch > 1 requires gM % tM == 0");
    static_assert((tilM & (tilM - 1)) == 0 && tilM >= 1 && tilM <= 128,
                  "tM must be pow2 in 1..128");
    static_assert((tilN & (tilN - 1)) == 0, "tN must be pow2");
    static_assert((tilK & (tilK - 1)) == 0, "tK must be pow2");

    static __half src0p[Batch * kMpad * globK + 2 * ALIGN];
    static __half src1p[Batch * globK * kNpad + 2 * ALIGN];
    static __half dstp[Batch * kMpad * kNpad + 2 * ALIGN];
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
#if PERF_NO_FILL
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
