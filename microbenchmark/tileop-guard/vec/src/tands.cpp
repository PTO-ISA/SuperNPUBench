#include "guard_case.hpp"
// TileOP-API doc guard: TANDS (VEC, tile-scalar, integer). Source: engines.md.
// Precision: res_check, scalar=5.
GUARD_SCALAR(int32_t, 16, 16, 5, [](auto& d, auto& s0, auto& sc){ TANDS(d, s0, sc); })
