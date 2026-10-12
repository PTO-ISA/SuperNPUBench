#pragma once

#include "fa_lowp.hpp"

namespace fa_lowp_recip {

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_lowp_recip_impl(
    __bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
    const __fp4_e2m1x2 *kPtr, const __fp4_e2m1x2 *vPtr,
    const __fp8_e8m0 *qScalePtr, const __fp8_e8m0 *kScalePtr,
    const __fp8_e8m0 *vScalePtr) {
    fa_lowp::flash_attention_lowp_impl<Sq, Skv, qD, vD, kTm, kTk,
                                       scaleD, true>(
        outPtr, qPtr, kPtr, vPtr, qScalePtr, kScalePtr, vScalePtr);
}

} // namespace fa_lowp_recip
