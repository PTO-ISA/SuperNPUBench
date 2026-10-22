#ifdef MATMUL_LOWP_REUSE_A
#include "basic_op/matmul/matmul_lowp_reuseA.hpp"
#else
#include "basic_op/matmul/matmul_shared_lowp.hpp"
#endif

#include <cstdint>
#include <cstring>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

#ifndef LOWP_DTYPE
#define LOWP_DTYPE __fp8_e4m3
#endif

#ifndef PACKED_FACTOR
#define PACKED_FACTOR 1
#endif

#ifndef USE_MX
#define USE_MX 0
#endif

#ifndef SCALE_DTYPE
#define SCALE_DTYPE __fp8_e8m0
#endif

#ifndef SCALE_GROUP
#define SCALE_GROUP 32
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

// M64 tail execution requires readable, zero-filled A/ScaleA rows. Keep
// logical M and C unchanged; only reuseA input allocations/batch strides pad.
#ifdef MATMUL_LOWP_REUSE_A
static constexpr int kInputM = (globM + 63) / 64 * 64;
#else
static constexpr int kInputM = globM;
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

using dtype = LOWP_DTYPE;
using scale_dtype = SCALE_DTYPE;

struct LowpMatmulContext {
    dtype *src0;
    dtype *src1;
    scale_dtype *src0Scale;
    scale_dtype *src1Scale;
    float *dst;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    LowpMatmulContext *context =
        static_cast<LowpMatmulContext *>(opaque);
    constexpr int kPhysicalGK =
        USE_MX && SCALE_GROUP == 64
            ? ((globK + tilK - 1) / tilK) * tilK
            : globK;
    constexpr int kStoredGK = kPhysicalGK / PACKED_FACTOR;
    constexpr int kGlobalScaleK =
        (kPhysicalGK + SCALE_GROUP - 1) / SCALE_GROUP;

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
#ifdef MATMUL_LOWP_REUSE_A
        matmul_lowp_reuseA<dtype, scale_dtype, PACKED_FACTOR, USE_MX != 0,
                           SCALE_GROUP, globM, globN, globK, tilM, tilN, tilK,
                           MATMUL_LOWP_TAIL_SHARED != 0>(
#else
        matmul_shared_lowp<dtype, scale_dtype, PACKED_FACTOR, USE_MX != 0,
                           SCALE_GROUP,
                           globM, globN, globK, tilM, tilN, tilK>(
#endif
            context->dst + b * globM * globN,
            context->src0 + b * kInputM * kStoredGK,
            // All lowp B operands are B-major [N, K/PACKED_FACTOR].
            context->src1 + b * kStoredGK * globN,
            context->src0Scale + b * kInputM * kGlobalScaleK,
            // All lowp B scales are B-major [N, K/group].
            context->src1Scale + b * kGlobalScaleK * globN);
    }
    BENCHEND;
    return 0;
}

int main() {
    using dtype = LOWP_DTYPE;
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    // Packed FP4x2 formats store two logical K values in each C++ element.
    // Their scale tensors still use the logical K dimension: one byte per 32
    // FP4 values, matching HiF4_HiF4.cpp's physical-K / 16 convention.
    constexpr int kPhysicalGK =
        USE_MX && SCALE_GROUP == 64
            ? ((globK + tilK - 1) / tilK) * tilK
            : globK;
    constexpr int kStoredGK = kPhysicalGK / PACKED_FACTOR;
    constexpr int kGlobalScaleK =
        (kPhysicalGK + SCALE_GROUP - 1) / SCALE_GROUP;

    static dtype src0p[Batch * kInputM * kStoredGK + 2 * ALIGN];
    // All lowp modes use B-major [N, K/PACKED_FACTOR].
    static dtype src1p[Batch * kStoredGK * globN + 2 * ALIGN];
    static scale_dtype
        src0Scalep[Batch * kInputM * kGlobalScaleK + 2 * ALIGN];
    static scale_dtype
        // B scale uses B-major [N, K/group].
        src1Scalep[Batch * kGlobalScaleK * globN + 2 * ALIGN];
    static float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 =
        (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 =
        (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    scale_dtype *src0Scale =
        (scale_dtype *)(((uint64_t)src0Scalep & ALIGN_MASK) + ALIGN);
    scale_dtype *src1Scale =
        (scale_dtype *)(((uint64_t)src1Scalep & ALIGN_MASK) + ALIGN);
    float *dst =
        (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == kIoTid) {
        readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                       Batch * globM * kStoredGK * sizeof(dtype));
        readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                       Batch * kStoredGK * globN * sizeof(dtype));
#if USE_MX
#define SRC0_SCALE_PATH CHK_DIR "/src0_scale.bin"
#define SRC1_SCALE_PATH CHK_DIR "/src1_scale.bin"
        readBinaryFile(SRC0_SCALE_PATH, (uint8_t *)src0Scale,
                       Batch * globM * kGlobalScaleK *
                           sizeof(scale_dtype));
        readBinaryFile(SRC1_SCALE_PATH, (uint8_t *)src1Scale,
                       Batch * kGlobalScaleK * globN *
                           sizeof(scale_dtype));
#endif
        // Files contain logical M rows per batch. Expand backwards so a
        // padded batch cannot overwrite the following batch's original data.
        if constexpr (kInputM != globM) {
            for (int b = Batch - 1; b >= 0; --b) {
                std::memmove(src0 + b * kInputM * kStoredGK,
                             src0 + b * globM * kStoredGK,
                             globM * kStoredGK * sizeof(dtype));
                std::memset(src0 + (b * kInputM + globM) * kStoredGK, 0,
                            (kInputM - globM) * kStoredGK * sizeof(dtype));
#if USE_MX
                std::memmove(src0Scale + b * kInputM * kGlobalScaleK,
                             src0Scale + b * globM * kGlobalScaleK,
                             globM * kGlobalScaleK * sizeof(scale_dtype));
                std::memset(src0Scale + (b * kInputM + globM) * kGlobalScaleK, 0,
                            (kInputM - globM) * kGlobalScaleK * sizeof(scale_dtype));
#endif
            }
        }
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#endif

    LowpMatmulContext context{src0, src1, src0Scale, src1Scale, dst};
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
    if (tid == kIoTid) {
        writeBinaryFile(RES_PATH, (uint8_t *)dst,
                        Batch * globM * globN * sizeof(float));
    }
#endif

    return status;
}
