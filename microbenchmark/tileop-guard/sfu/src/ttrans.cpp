#include "guard_case.hpp"
// TileOP-API doc guard: TTRANS (SFU layout) — transpose.
// Source: layout.md lists TTRANS as SFU op, no signature; (dst,src) inferred.
// Square 16x16 avoids NxM shape ambiguity. Precision: res_check, golden = src^T.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TTRANS(d, s); })
