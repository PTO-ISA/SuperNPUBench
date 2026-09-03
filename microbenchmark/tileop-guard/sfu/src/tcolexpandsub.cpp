#include "guard_case.hpp"
// TileOP-API doc guard: TCOLEXPANDSUB (SFU expand-arith, col broadcast 1×N).
// Semantics (docs/intrinsics/tcolexpandsub.md): dst[i,j] = f(src0[i,j], src1[0,j]);
// EXPDIF = exp(src0-src1). Precision: res_check, independent numpy golden.
GUARD_COLEXPAND(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TCOLEXPANDSUB(d, s0, s1); })
