#include "guard_case.hpp"
// TileOP-API doc guard: TLOG (SFU transcendental, elementwise natural log).
// Source: engines.md (no signature; (dst,src) unary inferred). Precision:
// res_check, independent numpy golden = log(x), positive domain, SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TLOG(d, s); })
