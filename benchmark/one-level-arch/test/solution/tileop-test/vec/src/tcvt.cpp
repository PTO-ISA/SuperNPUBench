#include "guard_case.hpp"
// TileOP-API doc guard: TCVT (VEC, elementwise numeric conversion fp32 -> s32).
// Source: docs/tileop-usage/reinterpret-tile.md — contrasts TCVT (numeric
// conversion, emits hardware op) vs reinterpret_tile (bit view). Signature
// shown there: TCVT(converted, src) with distinct dst dtype.
// Precision: res_check, host-generated fp32 (both signs, .5 midpoints), golden
// pins the rounding mode (see golden.py fam='cvt' round=...).
GUARD_UNARY_CVT(float, int32_t, 16, 16, [](auto& d, auto& s){ TCVT(d, s); })
