#include "multi_thread/fa/fa_2d_unroll_gmma.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

// MATRIX_DTYPE controls the matrix-resident Q/K/V inputs. VECTOR_DTYPE
// controls softmax, P, online accumulation and O. Both TMATMUL outputs are
// FP32 and are converted to VECTOR_DTYPE before vector processing.
// PackedFactor is 2 for the
// packed FP4x2 formats and 1 otherwise. MX/HIF modes additionally consume
// one E8M0 scale byte per 32 logical elements along each matmul K dimension.
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

#ifndef FA_X_DIM
#define FA_X_DIM 1
#endif

#ifndef FA_Y_DIM
#define FA_Y_DIM 1
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
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    constexpr int kStoredQD = qD / PACKED_FACTOR;
    constexpr int kStoredSkv = globSkv / PACKED_FACTOR;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kStoredTk = kTk / PACKED_FACTOR;
    constexpr int kTkScaleRows = (kStoredTk + 31) / 32;

    static_assert(globSq % kPeNum == 0,
                  "global Sq must be divisible by the PE count");

    static matrix_dtype qp[B * H * globSq * kStoredQD + 2 * ALIGN];
    static matrix_dtype kp[B * H * globSkv * kStoredQD + 2 * ALIGN];
    static matrix_dtype vp[B * H * kStoredSkv * vD + 2 * ALIGN];
    static vector_dtype outp[B * H * globSq * vD + 2 * ALIGN];
    static uint8_t qsp[B * H * globSq * (qD / 32) + 2 * ALIGN];
    static uint8_t ksp[B * H * globSkv * (qD / 32) + 2 * ALIGN];
    static uint8_t vsp[B * H * (globSkv / 32) * vD + 2 * ALIGN];
    static float prob_convertp[kPeNum * kPeTm * kTk + 2 * ALIGN];
    static matrix_dtype probp[kGroupM * kStoredTk + 2 * ALIGN];
    static uint8_t probsp[kGroupM * kTkScaleRows + 2 * ALIGN];
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
    float *prob_convert =
        (float *)(((uint64_t)prob_convertp & ALIGN_MASK) + ALIGN);
    matrix_dtype *prob =
        (matrix_dtype *)(((uint64_t)probp & ALIGN_MASK) + ALIGN);
    uint8_t *prob_scale =
        (uint8_t *)(((uint64_t)probsp & ALIGN_MASK) + ALIGN);

    // MX probability tiles are produced by an explicit vector-to-matrix
    // conversion, so their E8M0 scale is unity. Initialize the shared scale
    // scratch as raw bytes; constructing an E8M0 scalar in inline assembly is
    // not legal in the current backend.
    if (tid == kIoTid) {
        for (int idx = 0; idx < kGroupM * kTkScaleRows; ++idx) {
            prob_scale[idx] = 127;
        }
    }

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
            // PE0 loads the full shared Q/K/V tiles (PEMask=1), which are
            // then consumed cooperatively by all four PEs. The RES_CHECK I/O
            // uses the same PE0 thread.
            flash_attention_2d_unroll_shared_impl<
                matrix_dtype, vector_dtype, PACKED_FACTOR, USE_MX != 0,
                globSq, globSkv, qD, vD, kTm, kTk, FA_X_DIM, FA_Y_DIM>(
                out + i * H * globSq * vD + j * globSq * vD,
                q + i * H * globSq * kStoredQD + j * globSq * kStoredQD,
                k + i * H * globSkv * kStoredQD +
                    j * globSkv * kStoredQD,
                v + i * H * kStoredSkv * vD + j * kStoredSkv * vD,
                q_scale + i * H * globSq * (qD / 32) +
                    j * globSq * (qD / 32),
                k_scale + i * H * globSkv * (qD / 32) +
                    j * globSkv * (qD / 32),
                v_scale + i * H * (globSkv / 32) * vD +
                    j * (globSkv / 32) * vD,
                prob_convert + tid * kPeTm * kTk,
                prob, prob_scale);
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
