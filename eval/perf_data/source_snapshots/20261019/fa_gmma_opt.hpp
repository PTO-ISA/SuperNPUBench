#pragma once

#include "fa_gmma_kchains.hpp"

#ifndef FA_GMMA_OPT_ADDR_PRELOAD
#define FA_GMMA_OPT_ADDR_PRELOAD 1
#endif

static_assert(FA_GMMA_OPT_ADDR_PRELOAD == 0 || FA_GMMA_OPT_ADDR_PRELOAD == 1,
              "FA_GMMA_OPT_ADDR_PRELOAD must be 0 or 1");

// Optimized K-chain FlashAttention.
//
// Scheduling, online-softmax recurrence, Q reuse, and the persistent CUBE
// accumulator and fixpipe QK row-max come directly from fa_gmma_kchains.
// The last template argument enables K/V address lookahead.
template <typename MatrixDType, typename VectorDType,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_gmma_opt_impl(
    VectorDType *outPtr, MatrixDType *qPtr, MatrixDType *kPtr,
    MatrixDType *vPtr) {
    flash_attention_gmma_kchains_impl<
        MatrixDType, VectorDType, Sq, Skv, qD, vD, kTm,
        kTk, scaleD, FA_GMMA_OPT_ADDR_PRELOAD != 0>(
        outPtr, qPtr, kPtr, vPtr);
}

// K-tile lookahead variant: overlap K[j+1] TLOAD with PV[j] TMATMUL.
template <typename MatrixDType, typename VectorDType,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_gmma_opt_preload_impl(
    VectorDType *outPtr, MatrixDType *qPtr, MatrixDType *kPtr,
    MatrixDType *vPtr) {
    flash_attention_gmma_kchains_impl<
        MatrixDType, VectorDType, Sq, Skv, qD, vD, kTm,
        kTk, scaleD, FA_GMMA_OPT_ADDR_PRELOAD != 0, true>(
        outPtr, qPtr, kPtr, vPtr);
}

template <typename DType, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_gmma_opt_pto(DType *outPtr, DType *qPtr,
                                  DType *kPtr, DType *vPtr) {
    flash_attention_gmma_opt_impl<
        DType, DType, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        outPtr, qPtr, kPtr, vPtr);
}
