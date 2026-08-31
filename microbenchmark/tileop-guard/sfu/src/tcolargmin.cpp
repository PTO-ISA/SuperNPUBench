#include "guard_common.hpp"
// TileOP-API doc guard: TCOLARGMIN (SFU, reduce-and-expand)
// Source: docs/tileop-usage/engines.md — SFU | TCOLARGMIN | reduce-and-expand.
// NOTE(doc-gap): docs give NO signature/output shape. Col-reduce to 1 x N inferred.
int main() {
    constexpr int M = 16, N = 16;
    float a[M*N], c[1*N];
    gfill_seq(a, M*N, 1.0f); gzero(c, 1*N);
    BENCHSTART;
    g_colreduce<float, M, N>(c, a, [](auto& d, auto& s){ TCOLARGMIN(d, s); });
    BENCHEND;
    return 0;
}
