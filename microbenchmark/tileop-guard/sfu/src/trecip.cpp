#include "guard_case.hpp"
// TileOP-API doc guard: TRECIP (SFU transcendental, elementwise reciprocal).
// Source: engines.md (no signature; (dst,src) unary inferred). Precision:
// res_check, independent numpy golden = 1/x, nonzero domain, SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TRECIP(d, s); })
