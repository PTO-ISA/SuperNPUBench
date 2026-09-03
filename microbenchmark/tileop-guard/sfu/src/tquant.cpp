#include "guard_case.hpp"
// TileOP-API doc guard: TQUANT (SFU) — FP32 -> S8, RNE + saturate.
// Source: quant-and-im2col.md full signature TQUANT<Mode,Sat>(dst,src,mult,zp).
// Spec semantics: q = clamp(round_RNE(src*multiplier) + zeroPoint, -128, 127).
// GAP (found during golden dev, filed as gfrun issue): the emulator IGNORES the
// multiplier and zeroPoint args — output equals clamp(round_RNE(src)) regardless
// (verified 0/2048 mismatch at mult=1,zp=0; any non-identity mult/zp silently
// dropped). So this demo pins the CORE it does honor — RNE rounding + S8
// saturation (input spans +-256 to exercise both clamp edges) — with identity
// mult/zp; the ignored-scale path is tracked separately, not asserted here.
GUARD_UNARY_CVT(float, int8_t, 8, 256, [](auto& d, auto& s){
    TQUANT<RoundMode::RNE, /*saturate=*/true>(d, s, 1.0f, 0);
})
