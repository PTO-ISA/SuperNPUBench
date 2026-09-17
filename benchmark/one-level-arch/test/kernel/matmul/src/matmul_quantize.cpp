#include "basic_op/matmul/matmul_quantize.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

#ifndef DTYPE
#define DTYPE float
#endif

#ifndef OUT_DTYPE
#define OUT_DTYPE __fp8_e4m3
#endif

#ifndef USE_ASSEMBLE
#define USE_ASSEMBLE 1
#endif

#ifndef globM
#define globM 128
#endif

#ifndef globN
#define globN 64
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
#define tilK 128
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)
#define kBlock 32
#define kScaleCols (globN / kBlock)

using dtype = DTYPE;
using out_dtype = OUT_DTYPE;

// Phase 1 (matmul) writes FP32 C to cScratch; Phase 2 (quantize) reads it
// back and writes the packed data + per-row E8M0 scale. cScratch is an
// internal scratch buffer (not a kernel output).
struct MatmulQuantizeContext {
    dtype *src0;
    dtype *src1;
    float *cScratch;
    out_dtype *data;
    uint8_t *scale;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    MatmulQuantizeContext *context =
        static_cast<MatmulQuantizeContext *>(opaque);

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_quantize<dtype, globM, globN, globK, tilM, tilN, tilK,
                        USE_ASSEMBLE != 0, out_dtype>(
            context->data + b * globM * globN,
            context->scale + b * globM * kScaleCols,
            context->cScratch + b * globM * globN,
            context->src0 + b * globM * globK,
            context->src1 + b * globK * globN);
    }
    BENCHEND;
    return 0;
}

int main() {
    using dtype = DTYPE;
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();

    // A: [globM, globK], B: [globK, globN], C scratch: [globM, globN] FP32,
    // data: [globM, globN] OutT (FP4 packs 2/byte on TSTORE; both OutT kinds
    // are 1 byte elements, so globM*globN bytes covers the dense case),
    // scale: [globM, globN/32] E8M0 (one byte per MX block per row).
    static dtype src0p[Batch * globM * globK + 2 * ALIGN];
    static dtype src1p[Batch * globK * globN + 2 * ALIGN];
    static float cscratchp[Batch * globM * globN + 2 * ALIGN];
    static uint8_t datap[Batch * globM * globN + 2 * ALIGN];
    static uint8_t scalep[Batch * globM * kScaleCols + 2 * ALIGN];

    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *cScratch = (float *)(((uint64_t)cscratchp & ALIGN_MASK) + ALIGN);
    out_dtype *data =
        (out_dtype *)(((uint64_t)datap & ALIGN_MASK) + ALIGN);
    uint8_t *scale =
        (uint8_t *)(((uint64_t)scalep & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == kIoTid) {
        readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                       Batch * globM * globK * sizeof(dtype));
        readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                       Batch * globK * globN * sizeof(dtype));
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#endif

    MatmulQuantizeContext context{src0, src1, cScratch, data, scale};
#ifdef LINX_GROUP_RUNTIME
    const int status = linx_group_run(&context);
#else
    const int status = __linx_group_worker_main(0, &context);
#endif

#ifdef RES_CHECK
#define DATA_PATH CHK_DIR "/data.bin"
#define SCALE_PATH CHK_DIR "/scale.bin"
#ifndef LINX_GROUP_RUNTIME
    res_check_wait_for_all(res_check_sync, tid);
#endif
    if (tid == kIoTid) {
        // data: OutT is 1 byte (E4M3 dense, or FP4 packed 2/byte); the kernel
        // writes globM*globN OutT elements. For E4M3 that is globM*globN
        // bytes; for FP4 the valid footprint is globM*globN/2 bytes (the
        // trailing half is scratch and ignored by a golden comparator).
        writeBinaryFile(DATA_PATH, (uint8_t *)data,
                        Batch * globM * globN * sizeof(out_dtype));
        writeBinaryFile(SCALE_PATH, (uint8_t *)scale,
                        Batch * globM * kScaleCols * sizeof(uint8_t));
    }
#endif

    return status;
}
