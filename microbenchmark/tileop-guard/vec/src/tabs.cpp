#include "guard_common.hpp"
// TileOP-API doc guard: TABS (VEC, elementwise-tile-tile, unary)
// Source: docs/tileop-usage/engines.md — VEC | TABS | elementwise-tile-tile.
// NOTE(doc-gap): no signature in docs; (dst,src) inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE, -8.0f); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TABS(d, s); });
    BENCHEND;
    return 0;
}
