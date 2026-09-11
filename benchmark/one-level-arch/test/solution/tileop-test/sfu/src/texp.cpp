#include "guard_case.hpp"
// TileOP-API doc guard: TEXP (SFU transcendental, elementwise exp).
// Source: per-op 文档（早期 engines.md 无签名，现已补；(dst,src) unary inferred). Precision:
// res_check, independent numpy golden = exp(x) with SFU relative tolerance.
GUARD_UNARY(float, 16, 16, [](auto& d, auto& s){ TEXP(d, s); })
