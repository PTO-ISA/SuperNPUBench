#pragma once

#include "fa_2d_unroll_gmma.hpp"

// Four-PE FlashAttention with direct GM -> Shared TLOAD for K and V.
//
// Q remains a PE-private left tile. K and V are first described as local
// TileRight shapes and then wrapped by SharedTile, matching the loading path in
// kernels/matmul/matmul_shared.hpp:
//
//   Global K/V -> TLOAD -> SharedTile<TileRight<...>> -> TMATMUL_FIXP
//
// The online-softmax state and output tiles remain PE-private vector tiles.
template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_matmul_shared(dtype *out_ptr, dtype *q_ptr,
                                   dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        true, dtype, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}
