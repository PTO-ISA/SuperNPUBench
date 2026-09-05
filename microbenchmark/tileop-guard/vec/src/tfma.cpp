#include "guard_case.hpp"
// TileOP-API doc guard: TFMA (VEC, ternary a*b+c). Source: engines.md (no signature;
// (dst,a,b,c) inferred). Precision: res_check.
GUARD_TERNARY(float, 16, 16, [](auto& d, auto& s0, auto& s1, auto& s2){ TFMA(d, s0, s1, s2); })
