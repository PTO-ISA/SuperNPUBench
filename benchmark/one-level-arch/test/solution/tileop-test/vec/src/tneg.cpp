#include "guard_case.hpp"
// TileOP-API doc guard: TNEG (VEC, elementwise-tile-tile, unary)
// Source: per-op 文档（早期 engines.md 无签名，现已补；(dst,src) inferred). Precision: res_check.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TNEG(d, s); })
