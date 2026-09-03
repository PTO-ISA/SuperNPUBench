#include "guard_case.hpp"
// TileOP-API doc guard: TEXP (SFU transcendental, elementwise exp).
// Source: engines.md (no signature; (dst,src) unary inferred). Precision:
// res_check, independent numpy golden = exp(x) with SFU relative tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TEXP(d, s); })
