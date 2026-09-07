#include "guard_case.hpp"
// TileOP-API doc guard: TPARTMIN (SFU irregular) — elementwise min over the
// common valid region. "PART" = partial-valid-region (docs/intrinsics/tpartmin.md),
// NOT a segmented reduction. Full-valid 16x16 => plain elementwise min.
// Precision: res_check, independent numpy golden.
GUARD_BINARY(float, 16, 16, [](auto& d, auto& s0, auto& s1){ TPARTMIN(d, s0, s1); })
