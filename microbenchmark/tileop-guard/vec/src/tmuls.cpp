#include "guard_case.hpp"
// TileOP-API doc guard: TMULS (VEC, tile-scalar). Source: engines.md (no signature).
// Precision: res_check, scalar=1.75.
GUARD_SCALAR(float, 16, 16, 1.75f, [](auto& d, auto& s0, auto& sc){ TMULS(d, s0, sc); })
