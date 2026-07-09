#ifndef FA_2D_UNROLL_HPP
#define FA_2D_UNROLL_HPP

#include "fa_2d_unroll_pto.hpp"

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void flash_attention_2d_unroll(dtype* out_ptr, dtype* q_ptr, dtype* k_ptr, dtype* v_ptr) {
    flash_attention_2d_unroll_pto<dtype, Sq, Skv, qD, vD, kTm, kTk, qD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}

#endif
