#include "guard_case.hpp"
// TileOP-API doc guard: TROWPROD (SFU reduce, row-reduce over cols).
// Source: engines.md (no signature/output-shape). Output physical MxN, valid Mx1
// at out[r*N+0]. Precision: res_check, independent numpy golden.
GUARD_ROWREDUCE(float, 16, 16, [](auto& d, auto& s){ TROWPROD(d, s); })
