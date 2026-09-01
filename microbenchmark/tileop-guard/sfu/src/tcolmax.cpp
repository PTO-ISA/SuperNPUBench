#include "guard_case.hpp"
// TileOP-API doc guard: TCOLMAX (SFU reduce, col-reduce over rows).
// Source: engines.md (no signature/output-shape). Output physical MxN, valid 1xN
// at out[0*N+c]. Precision: res_check, independent numpy golden.
GUARD_COLREDUCE(float, 16, 16, [](auto& d, auto& s){ TCOLMAX(d, s); })
