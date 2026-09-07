#include "guard_case.hpp"
// TileOP-API doc guard: TADDS (VEC, tile-scalar). Source: per-op 文档（早期 engines.md 无签名，现已补）.
// Precision: res_check, scalar=1.75.
GUARD_SCALAR(float, 16, 16, 1.75f, [](auto& d, auto& s0, auto& sc){ TADDS(d, s0, sc); })
