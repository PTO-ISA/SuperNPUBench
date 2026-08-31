#include "guard_common.hpp"
// TileOP-API doc guard: TCOLEXPAND (SFU, reduce-and-expand)
// Source: docs/tileop-usage/engines.md — SFU | TCOLEXPAND | reduce-and-expand.
// NOTE(doc-gap): docs give NO signature. Expand back to M x N inferred (unary).
int main() {
    constexpr int M = 16, N = 16, NE = M*N;
    float a[NE], c[NE];
    gfill_seq(a, NE, 1.0f); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TCOLEXPAND(d, s); });
    BENCHEND;
    return 0;
}
