#include "guard_case.hpp"
// TileOP-API doc guard: TSQRT (SFU transcendental, elementwise square root).
// Source: engines.md (no signature; (dst,src) unary inferred). Precision:
// res_check, independent numpy golden = sqrt(x), nonneg domain, SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TSQRT(d, s); })
