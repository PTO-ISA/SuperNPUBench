#include "matmul/matmul_shared.hpp"

#include <cstdint>
#include <unistd.h>

#include "benchmark.h"
#include "fileop.h"

// Element data type for the A/B input tiles. Set via -DDTYPE=<token> from the
// Makefile (float / __bf16 / __half). The output C tile stays FP32 inside the
// kernel template (see matmul_shared.hpp).
#ifndef DTYPE
#define DTYPE float
#endif

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
#define tilM 32
#endif

#ifndef tilN
#define tilN 32
#endif

#ifndef tilK
#define tilK 32
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    using dtype = DTYPE;
    constexpr int kPeNum = 4;
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();

    static_assert(globM % kPeNum == 0,
                  "global M must be divisible by the PE count");

    static dtype src0p[Batch * globM * globK + 2 * ALIGN];
    static dtype src1p[Batch * globK * globN + 2 * ALIGN];
    static float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static volatile int leader_ready = 0;
    if (tid == kIoTid) {
        readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                       Batch * globM * globK * sizeof(dtype));
        readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                       Batch * globK * globN * sizeof(dtype));
        __asm__ volatile("" : : : "memory");
        leader_ready = 1;
    } else {
        while (!leader_ready) {
        }
        __asm__ volatile("" : : : "memory");
    }
#endif

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_shared<dtype, globM, globN, globK,
                      tilM, tilN, tilK>(
            dst + b * globM * globN,
            src0 + b * globM * globK,
            src1 + b * globK * globN);
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    if (tid == kIoTid) {
        writeBinaryFile(RES_PATH, (uint8_t *)dst,
                        Batch * globM * globN * sizeof(float));
    }
#endif

    return 0;
}
