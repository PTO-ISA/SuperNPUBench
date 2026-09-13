#include "guard_case.hpp"
// TileOP-API doc guard: TCOLMIN (SFU reduce, col-reduce over rows).
// Source: per-op 文档（早期 engines.md 无签名/输出形状，现已补）. Output physical MxN, valid 1xN
// at out[0*N+c]. Precision: res_check, independent numpy golden.
GUARD_COLREDUCE(float, 16, 16, [](auto& d, auto& s){ TCOLMIN(d, s); })
