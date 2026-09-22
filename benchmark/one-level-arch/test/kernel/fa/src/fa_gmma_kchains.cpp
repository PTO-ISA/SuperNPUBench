#include "basic_op/fa/fa_gmma_kchains.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#ifndef MATRIX_DTYPE
#define MATRIX_DTYPE float
#endif

#ifndef VECTOR_DTYPE
#define VECTOR_DTYPE float
#endif

#ifndef PACKED_FACTOR
#define PACKED_FACTOR 1
#endif

#ifndef Tsq
#define globSq 512
#else
#define globSq Tsq
#endif

#ifndef Tskv
#define globSkv 512
#else
#define globSkv Tskv
#endif

#ifndef FA_QD
#define FA_QD 128
#endif

#ifndef FA_VD
#define FA_VD 128
#endif

#ifndef Tm
#define kTm 128
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 128
#else
#define kTk Tk
#endif

#define B 1
#define H 1
#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    using MatrixDType = MATRIX_DTYPE;
    using VectorDType = VECTOR_DTYPE;
    constexpr int kIoTid = 0;
    constexpr int kStoredQD = FA_QD / PACKED_FACTOR;
    constexpr int kStoredSkv = globSkv / PACKED_FACTOR;
    const uint32_t tid = get_thread_idx();

    static MatrixDType qp[B * H * globSq * kStoredQD + 2 * ALIGN];
    static MatrixDType kp[B * H * globSkv * kStoredQD + 2 * ALIGN];
    static MatrixDType vp[B * H * kStoredSkv * FA_VD + 2 * ALIGN];
    static VectorDType outp[B * H * globSq * FA_VD + 2 * ALIGN];
#ifdef RES_CHECK
    static MultiThreadResCheckSync resCheckSync{};
#endif

    auto *q = reinterpret_cast<MatrixDType *>(
        (reinterpret_cast<uint64_t>(qp) & ALIGN_MASK) + ALIGN);
    auto *k = reinterpret_cast<MatrixDType *>(
        (reinterpret_cast<uint64_t>(kp) & ALIGN_MASK) + ALIGN);
    auto *v = reinterpret_cast<MatrixDType *>(
        (reinterpret_cast<uint64_t>(vp) & ALIGN_MASK) + ALIGN);
    auto *out = reinterpret_cast<VectorDType *>(
        (reinterpret_cast<uint64_t>(outp) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    if (tid == kIoTid) {
        readBinaryFile(CHK_DIR "/srcq.bin", reinterpret_cast<uint8_t *>(q),
                       B * H * globSq * kStoredQD * sizeof(MatrixDType));
        readBinaryFile(CHK_DIR "/srck.bin", reinterpret_cast<uint8_t *>(k),
                       B * H * globSkv * kStoredQD * sizeof(MatrixDType));
        readBinaryFile(CHK_DIR "/srcv.bin", reinterpret_cast<uint8_t *>(v),
                       B * H * kStoredSkv * FA_VD * sizeof(MatrixDType));
    }
    res_check_publish_inputs(resCheckSync, tid);
#endif

    BENCHSTART;
    flash_attention_gmma_kchains_impl<
        MatrixDType, VectorDType, PACKED_FACTOR, globSq, globSkv,
        FA_QD, FA_VD, kTm, kTk>(out, q, k, v);
    BENCHEND;

#ifdef RES_CHECK
    res_check_wait_for_all(resCheckSync, tid);
    if (tid == kIoTid) {
        writeBinaryFile(CHK_DIR "/res.bin", reinterpret_cast<uint8_t *>(out),
                        B * H * globSq * FA_VD * sizeof(VectorDType));
    }
#endif

    return 0;
}
