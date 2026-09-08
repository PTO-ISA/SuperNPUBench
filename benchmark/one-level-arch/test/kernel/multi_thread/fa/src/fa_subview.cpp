#include "multi_thread/fa/fa_subview.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

// Subview FA driver — same I/O contract as fa_2d_unroll_gmma.cpp but includes the
// fa_subview.hpp kernel (TPARTVIEW column sub-views for TROWMAX/TROWSUM
// to satisfy the 2048-byte reduction-source ISA limit).

#ifndef MATRIX_DTYPE
#define MATRIX_DTYPE float
#endif

#ifndef VECTOR_DTYPE
#define VECTOR_DTYPE float
#endif

#ifndef PACKED_FACTOR
#define PACKED_FACTOR 1
#endif

#ifndef USE_MX
#define USE_MX 0
#endif

#define B 1
#define H 1

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

#ifndef qD
#ifdef FA_QD
#define qD FA_QD
#else
#define qD 128
#endif
#endif

#ifndef vD
#ifdef FA_VD
#define vD FA_VD
#else
#define vD 128
#endif
#endif

#ifndef Tm
#define kTm 16
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 16
#else
#define kTk Tk
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    using matrix_dtype = MATRIX_DTYPE;
    using vector_dtype = VECTOR_DTYPE;
    constexpr int kPeNum = 4;
    constexpr int kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    constexpr int kStoredQD = qD / PACKED_FACTOR;
    constexpr int kStoredSkv = globSkv / PACKED_FACTOR;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kStoredTk = kTk / PACKED_FACTOR;

    static_assert(globSq % kPeNum == 0,
                  "global Sq must be divisible by the PE count");

    static matrix_dtype qp[B * H * globSq * kStoredQD + 2 * ALIGN];
    static matrix_dtype kp[B * H * globSkv * kStoredQD + 2 * ALIGN];
    static matrix_dtype vp[B * H * kStoredSkv * vD + 2 * ALIGN];
    static vector_dtype outp[B * H * globSq * vD + 2 * ALIGN];
    static uint8_t qsp[B * H * globSq * (qD / 32) + 2 * ALIGN];
    static uint8_t ksp[B * H * globSkv * (qD / 32) + 2 * ALIGN];
    static uint8_t vsp[B * H * (globSkv / 32) * vD + 2 * ALIGN];
#ifdef RES_CHECK
    static MultiThreadResCheckSync res_check_sync{};
#endif

    matrix_dtype *q =
        (matrix_dtype *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    matrix_dtype *k =
        (matrix_dtype *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    matrix_dtype *v =
        (matrix_dtype *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    vector_dtype *out =
        (vector_dtype *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);
    uint8_t *q_scale = (uint8_t *)(((uint64_t)qsp & ALIGN_MASK) + ALIGN);
    uint8_t *k_scale = (uint8_t *)(((uint64_t)ksp & ALIGN_MASK) + ALIGN);
    uint8_t *v_scale = (uint8_t *)(((uint64_t)vsp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRCQ_PATH CHK_DIR "/srcq.bin"
#define SRCK_PATH CHK_DIR "/srck.bin"
#define SRCV_PATH CHK_DIR "/srcv.bin"
    if (tid == kIoTid) {
        readBinaryFile(SRCQ_PATH, (uint8_t *)q,
                       B * H * globSq * kStoredQD * sizeof(matrix_dtype));
        readBinaryFile(SRCK_PATH, (uint8_t *)k,
                       B * H * globSkv * kStoredQD * sizeof(matrix_dtype));
        readBinaryFile(SRCV_PATH, (uint8_t *)v,
                       B * H * kStoredSkv * vD * sizeof(matrix_dtype));
#if USE_MX
#define SRCQS_PATH CHK_DIR "/srcq_scale.bin"
#define SRCKS_PATH CHK_DIR "/srck_scale.bin"
#define SRCVS_PATH CHK_DIR "/srcv_scale.bin"
        readBinaryFile(SRCQS_PATH, q_scale,
                       B * H * globSq * (qD / 32));
        readBinaryFile(SRCKS_PATH, k_scale,
                       B * H * globSkv * (qD / 32));
        readBinaryFile(SRCVS_PATH, v_scale,
                       B * H * (globSkv / 32) * vD);
#endif
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif

    BENCHSTART;
#pragma clang loop unroll(full)
    for (int i = 0; i < B; ++i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < H; ++j) {
            flash_attention_2d_unroll_shared_impl<
                matrix_dtype, vector_dtype, PACKED_FACTOR,
                globSq, globSkv, qD, vD, kTm, kTk>(
                out + i * H * globSq * vD + j * globSq * vD,
                q + i * H * globSq * kStoredQD + j * globSq * kStoredQD,
                k + i * H * globSkv * kStoredQD +
                    j * globSkv * kStoredQD,
                v + i * H * kStoredSkv * vD + j * kStoredSkv * vD);
        }
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == kIoTid) {
        writeBinaryFile(RES_PATH, (uint8_t *)out,
                        B * H * globSq * vD * sizeof(vector_dtype));
    }
#endif

    return 0;
}
