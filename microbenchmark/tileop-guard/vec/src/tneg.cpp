#include "guard_case.hpp"
// TileOP-API doc guard: TNEG (VEC, elementwise-tile-tile, unary)
// Source: engines.md (no signature; (dst,src) inferred). Precision: res_check.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TNEG(d, s); })
