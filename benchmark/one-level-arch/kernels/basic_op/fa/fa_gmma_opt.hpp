#pragma once

#include "fa_gmma_kchains.hpp"

// Optimized K-chain FlashAttention.
//
// Scheduling, online-softmax recurrence, Q reuse, and the persistent CUBE
// accumulator come directly from fa_gmma_kchains. The last template argument
// enables the QK fixpipe row-max result and removes the standalone TROWMAX
// traversal over the score tile.
template <typename MatrixDType, typename VectorDType, int PackedFactor,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_gmma_opt_impl(
    VectorDType *outPtr, MatrixDType *qPtr, MatrixDType *kPtr,
    MatrixDType *vPtr) {
    flash_attention_gmma_kchains_impl<
        MatrixDType, VectorDType, PackedFactor, Sq, Skv, qD, vD, kTm,
        kTk, scaleD, true>(outPtr, qPtr, kPtr, vPtr);
}

template <typename DType, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_gmma_opt_pto(DType *outPtr, DType *qPtr,
                                  DType *kPtr, DType *vPtr) {
    flash_attention_gmma_opt_impl<
        DType, DType, 1, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        outPtr, qPtr, kPtr, vPtr);
}
