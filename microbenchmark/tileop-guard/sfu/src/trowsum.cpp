#include "guard_common.hpp"
// TileOP-API doc guard: TROWSUM (SFU, reduce-and-expand)
// Source: docs/tileop-usage/engines.md — SFU | TROWSUM | reduce-and-expand.
// NOTE(doc-gap): docs give NO signature and do not state the output shape.
// Row-reduce to M x 1 inferred (matches reference-tree ValidCol==1 rule).
int main() {
    constexpr int M = 16, N = 16;
    float a[M*N], c[M*1];
    gfill_seq(a, M*N, 1.0f); gzero(c, M*1);
    BENCHSTART;
    g_rowreduce<float, M, N>(c, a, [](auto& d, auto& s){ TROWSUM(d, s); });
    BENCHEND;
    return 0;
}
