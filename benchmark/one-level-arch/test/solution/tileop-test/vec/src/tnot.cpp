#include "guard_case.hpp"
// TileOP-API doc guard: TNOT (VEC, elementwise-tile-tile, unary integer).
// Source: engines.md. Precision: res_check.
GUARD_UNARY(int32_t, 16, 16, [](auto& d, auto& s){ TNOT(d, s); })
