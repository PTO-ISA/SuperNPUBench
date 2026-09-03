#include "guard_case.hpp"
// TileOP-API doc guard: TQUANT (SFU) — FP32 -> S8, RNE + saturate.
// Source: quant-and-im2col.md full signature TQUANT<Mode,Sat>(dst,src,mult,zp).
// Authoritative semantics (pto-spec normative ASL, format-conversion/TQUANT.md):
//   q = clamp(round_RNE(src*multiplier + zeroPoint), -128, 127); multiplier and
//   zeroPoint are NORMATIVELY MANDATORY (only an omitted B.IOR defaults to 1.0/0).
// So this demo passes REAL non-identity mult/zp (0.5f, 1) per spec, and the golden
// asserts the full quant. The emulator currently IGNORES mult/zp (output equals
// clamp(round_RNE(src))), so this case is expected to land in PRECISION-FAIL,
// witnessing model gap gfrun-5 (not a golden/demo regression). Input spans +-256.
GUARD_UNARY_CVT(float, int8_t, 8, 256, [](auto& d, auto& s){
    TQUANT<RoundMode::RNE, /*saturate=*/true>(d, s, 0.5f, 1);
})
