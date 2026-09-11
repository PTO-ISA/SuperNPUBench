#include "guard_case.hpp"
// TileOP-API doc guard: TRECIP (SFU transcendental, elementwise reciprocal).
// Source: per-op 文档（早期 engines.md 无签名，现已补；(dst,src) unary inferred). Precision:
// res_check, independent numpy golden = 1/x, nonzero domain, SFU tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TRECIP(d, s); })
