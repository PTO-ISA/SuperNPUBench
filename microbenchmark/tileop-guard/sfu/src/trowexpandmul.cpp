#include "guard_case.hpp"
// TileOP-API doc guard: TROWEXPANDMUL (SFU expand-arith, row broadcast M×1).
// Semantics (docs/intrinsics/trowexpandmul.md): dst[i,j] = f(src0[i,j], src1[i,0]);
// EXPDIF = exp(src0-src1). Precision: res_check, independent numpy golden.
GUARD_ROWEXPAND(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TROWEXPANDMUL(d, s0, s1); })
