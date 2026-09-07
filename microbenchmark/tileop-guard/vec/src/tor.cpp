#include "guard_case.hpp"
// TileOP-API doc guard: TOR (VEC, elementwise-tile-tile, integer binary)
// Source: per-op 文档（早期 engines.md 无签名/dtype，现已补；int32 chosen）. Precision: res_check.
GUARD_BINARY(int32_t, 16, 16, [](auto& d, auto& s0, auto& s1){ TOR(d, s0, s1); })
