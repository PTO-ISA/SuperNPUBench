#include "guard_case.hpp"
// TileOP-API doc guard: TSUB (VEC, elementwise-tile-tile, binary)
// Source: engines.md (no C++ signature; dst-first inferred). Precision: res_check,
// host-owned inputs (golden.py), independent numpy golden.
GUARD_BINARY(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TSUB(d, s0, s1); })
