#include "matmul/matmul_shared_lowp.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

#ifndef LOWP_DTYPE
#define LOWP_DTYPE __fp8_e4m3
#endif

#ifndef PACKED_FACTOR
#define PACKED_FACTOR 1
#endif

#ifndef USE_MX
#define USE_MX 0
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

int main() {
    using dtype = LOWP_DTYPE;
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    // Packed FP4x2 formats store two logical K values in each C++ element.
    // Their scale tensors still use the logical K dimension: one byte per 32
    // FP4 values, matching HiF4_HiF4.cpp's physical-K / 16 convention.
    constexpr int kStoredGK = globK / PACKED_FACTOR;

    static dtype src0p[Batch * globM * kStoredGK + 2 * ALIGN];
    static dtype src1p[Batch * kStoredGK * globN + 2 * ALIGN];
    static uint8_t src0Scalep[Batch * globM * (globK / 32) + 2 * ALIGN];
    static uint8_t src1Scalep[Batch * (globK / 32) * globN + 2 * ALIGN];
    static float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 =
        (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 =
        (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    uint8_t *src0Scale =
        (uint8_t *)(((uint64_t)src0Scalep & ALIGN_MASK) + ALIGN);
    uint8_t *src1Scale =
        (uint8_t *)(((uint64_t)src1Scalep & ALIGN_MASK) + ALIGN);
    float *dst =
        (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static volatile int leader_ready = 0;
    if (tid == kIoTid) {
        readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                       Batch * globM * kStoredGK * sizeof(dtype));
        readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                       Batch * kStoredGK * globN * sizeof(dtype));
#if USE_MX
#define SRC0_SCALE_PATH CHK_DIR "/src0_scale.bin"
#define SRC1_SCALE_PATH CHK_DIR "/src1_scale.bin"
        readBinaryFile(SRC0_SCALE_PATH, src0Scale,
                       Batch * globM * (globK / 32));
        readBinaryFile(SRC1_SCALE_PATH, src1Scale,
                       Batch * (globK / 32) * globN);
#endif
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
        matmul_shared_lowp<dtype, PACKED_FACTOR, USE_MX != 0,
                           globM, globN, globK, tilM, tilN, tilK>(
            dst + b * globM * globN,
            src0 + b * globM * kStoredGK,
            src1 + b * kStoredGK * globN,
            src0Scale + b * globM * (globK / 32),
            src1Scale + b * (globK / 32) * globN);
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
