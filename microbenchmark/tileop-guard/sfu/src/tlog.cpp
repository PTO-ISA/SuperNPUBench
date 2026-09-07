#include "guard_case.hpp"
// TileOP-API doc guard: TLOG (SFU transcendental, elementwise natural log).
// Source: per-op 文档（早期 engines.md 无签名，现已补；(dst,src) unary inferred). Precision:
// res_check, independent numpy golden = log(x), positive domain, SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TLOG(d, s); })
