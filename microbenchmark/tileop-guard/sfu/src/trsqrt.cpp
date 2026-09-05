#include "guard_case.hpp"
// TileOP-API doc guard: TRSQRT (SFU transcendental, elementwise inverse sqrt).
// Source: engines.md (no signature; (dst,src) unary inferred; fp32-only per
// reference tree). Precision: res_check, numpy golden = 1/sqrt(x), SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TRSQRT(d, s); })
