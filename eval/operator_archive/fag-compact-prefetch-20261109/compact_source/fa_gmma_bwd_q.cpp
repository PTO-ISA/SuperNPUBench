// TileLang-style long-Q FAG driver: fixed KV tile (M=128), inner Q loop.
// Default: FP8 E4M3 Q/K/V/dO, head_dim=256; FP32 accumulators and lse/delta.
#include "basic_op/fa/fa_gmma_bwd_q.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#ifdef FA_BWD_CHECK_DIR
#undef CHK_DIR
#define CHK_DIR FA_BWD_CHECK_DIR
#endif

using namespace pto;

#ifndef MATRIX_DTYPE
#define MATRIX_DTYPE __fp8_e4m3
#endif
#ifndef VECTOR_DTYPE
#define VECTOR_DTYPE float
#endif
#ifndef Tsq
#define globSq 4096
#else
#define globSq Tsq
#endif
#ifndef Tskv
#define globSkv 256
#else
#define globSkv Tskv
#endif
#ifndef FA_QD
#define FA_QD 256
#endif
#ifndef FA_VD
#define FA_VD 256
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

template <typename T, int N>
static T *fa_bwd_q_align(T (&arr)[N]) {
    return reinterpret_cast<T *>(
        (reinterpret_cast<uint64_t>(arr) & ALIGN_MASK) + ALIGN);
}

int main() {
    using MatDType = MATRIX_DTYPE;
    using VecDType = VECTOR_DTYPE;
    constexpr int Sq = globSq, Skv = globSkv, qD = FA_QD, vD = FA_VD;
    constexpr int TkDim = kTk;
    constexpr int auxElements = FA_BWD_Q_COMPACT_AUX ? Sq : TkDim * Sq;
    constexpr int kKVBlocks = Skv / TkDim;
    const uint32_t tid = get_thread_idx();

    static MatDType qP[Sq * qD + 2 * ALIGN];
    static MatDType kP[Skv * qD + 2 * ALIGN];
    static MatDType vP[Skv * vD + 2 * ALIGN];
    static MatDType doP[Sq * vD + 2 * ALIGN];
    static VecDType lseP[auxElements + 2 * ALIGN];
    static VecDType deltaP[auxElements + 2 * ALIGN];
    // The kernel stages a distinct dS tile for every (KV block, Q block).
    static MatDType dsP[kKVBlocks * TkDim * Sq + 2 * ALIGN];
    // Static storage is zero-initialized before the first atomic update.
    static VecDType dqP[Sq * qD + 2 * ALIGN];
    static VecDType dkP[Skv * qD + 2 * ALIGN];
    static VecDType dvP[Skv * vD + 2 * ALIGN];
#ifdef RES_CHECK
    static MultiThreadResCheckSync resCheckSync{};
#endif

    auto *q = fa_bwd_q_align(qP);
    auto *k = fa_bwd_q_align(kP);
    auto *v = fa_bwd_q_align(vP);
    auto *dO = fa_bwd_q_align(doP);
    auto *lseB = fa_bwd_q_align(lseP);
    auto *deltaB = fa_bwd_q_align(deltaP);
    auto *dsStage = fa_bwd_q_align(dsP);
    auto *dQ = fa_bwd_q_align(dqP);
    auto *dK = fa_bwd_q_align(dkP);
    auto *dV = fa_bwd_q_align(dvP);

#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/srcq.bin", reinterpret_cast<uint8_t *>(q),
                       Sq * qD * sizeof(MatDType));
        readBinaryFile(CHK_DIR "/srck.bin", reinterpret_cast<uint8_t *>(k),
                       Skv * qD * sizeof(MatDType));
        readBinaryFile(CHK_DIR "/srcv.bin", reinterpret_cast<uint8_t *>(v),
                       Skv * vD * sizeof(MatDType));
        readBinaryFile(CHK_DIR "/srcdo.bin", reinterpret_cast<uint8_t *>(dO),
                       Sq * vD * sizeof(MatDType));
#if FA_BWD_Q_COMPACT_AUX
#define FA_BWD_Q_LSE_FILE "lse"
#define FA_BWD_Q_DELTA_FILE "delta"
#else
#define FA_BWD_Q_LSE_FILE "lseB"
#define FA_BWD_Q_DELTA_FILE "deltaB"
#endif
        readBinaryFile(CHK_DIR "/src" FA_BWD_Q_LSE_FILE ".bin", reinterpret_cast<uint8_t *>(lseB),
                       auxElements * sizeof(VecDType));
        readBinaryFile(CHK_DIR "/src" FA_BWD_Q_DELTA_FILE ".bin",
                       reinterpret_cast<uint8_t *>(deltaB),
                       auxElements * sizeof(VecDType));
        // Exact read-back guards in the host checker verify input visibility.
        writeBinaryFile(CHK_DIR "/readback_q.bin", reinterpret_cast<uint8_t *>(q),
                        Sq * qD * sizeof(MatDType));
        writeBinaryFile(CHK_DIR "/readback_k.bin", reinterpret_cast<uint8_t *>(k),
                        Skv * qD * sizeof(MatDType));
        writeBinaryFile(CHK_DIR "/readback_v.bin", reinterpret_cast<uint8_t *>(v),
                        Skv * vD * sizeof(MatDType));
        writeBinaryFile(CHK_DIR "/readback_do.bin", reinterpret_cast<uint8_t *>(dO),
                        Sq * vD * sizeof(MatDType));
        writeBinaryFile(CHK_DIR "/readback_" FA_BWD_Q_LSE_FILE ".bin", reinterpret_cast<uint8_t *>(lseB),
                        auxElements * sizeof(VecDType));
        writeBinaryFile(CHK_DIR "/readback_" FA_BWD_Q_DELTA_FILE ".bin", reinterpret_cast<uint8_t *>(deltaB),
                        auxElements * sizeof(VecDType));
    }
    res_check_publish_inputs(resCheckSync, tid);
#endif

    BENCHSTART;
    flash_attention_bwd_q_pto<MatDType, VecDType, Sq, Skv, qD, vD, TkDim, TkDim>(
        dQ, dK, dV, q, k, v, dO, lseB, deltaB, dsStage);
    BENCHEND;

#ifdef RES_CHECK
    res_check_wait_for_all(resCheckSync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/res_dq.bin",
                        reinterpret_cast<uint8_t *>(dQ),
                        Sq * qD * sizeof(VecDType));
        writeBinaryFile(CHK_DIR "/res_dk.bin", reinterpret_cast<uint8_t *>(dK),
                        Skv * qD * sizeof(VecDType));
        writeBinaryFile(CHK_DIR "/res_dv.bin", reinterpret_cast<uint8_t *>(dV),
                        Skv * vD * sizeof(VecDType));
    }
#endif
    return 0;
}
