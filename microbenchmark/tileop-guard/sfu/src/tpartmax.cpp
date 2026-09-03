#include "guard_case.hpp"
// TileOP-API doc guard: TPARTMAX (SFU irregular) — elementwise max over the
// common valid region. "PART" = partial-valid-region (docs/intrinsics/tpartmax.md),
// NOT a segmented reduction. Full-valid 16x16 => plain elementwise max.
// Precision: res_check, independent numpy golden.
GUARD_BINARY(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TPARTMAX(d, s0, s1); })
