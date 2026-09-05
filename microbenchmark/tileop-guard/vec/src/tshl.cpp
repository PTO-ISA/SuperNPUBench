#include "guard_case.hpp"
// TileOP-API doc guard: TSHL (VEC, elementwise-tile-tile, integer binary)
// Source: engines.md (no signature/dtype; int32 chosen). Precision: res_check.
GUARD_BINARY(int32_t, 16, 16, [](auto& d, auto& s0, auto& s1){ TSHL(d, s0, s1); })
