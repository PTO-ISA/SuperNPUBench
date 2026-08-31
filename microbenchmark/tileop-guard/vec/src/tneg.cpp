#include "guard_common.hpp"
// TileOP-API doc guard: TNEG (VEC, elementwise-tile-tile, unary)
// Source: docs/tileop-usage/engines.md — VEC | TNEG | elementwise-tile-tile.
// NOTE(doc-gap): no signature in docs; (dst,src) inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TNEG(d, s); });
    BENCHEND;
    return 0;
}
