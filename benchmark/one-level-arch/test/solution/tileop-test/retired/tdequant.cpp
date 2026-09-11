#include "guard_case.hpp"
// TileOP-API doc guard: TDEQUANT (SFU) — S8 -> FP32.
// Source: quant-and-im2col.md full signature TDEQUANT<Mode>(dst,src,mult,zp).
// Semantics: dst = (src - zeroPoint) * multiplier. mult=2.0, zp=0.
// Precision: res_check, independent numpy golden.
GUARD_UNARY_CVT(int8_t, float, 8, 256, [](auto& d, auto& s){
    TDEQUANT<RoundMode::RTZ>(d, s, 2.0f, 0);
})
