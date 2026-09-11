#include "guard_case.hpp"
// TileOP-API doc guard: TPARTMUL (SFU irregular) — elementwise mul over the
// common valid region. "PART" = partial-valid-region (docs/intrinsics/tpartmul.md),
// NOT a segmented reduction. Full-valid 16x16 => plain elementwise mul.
// Precision: res_check, independent numpy golden.
GUARD_BINARY(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TPARTMUL(d, s0, s1); })
