#include "guard_common.hpp"
// TileOP-API doc guard: TNOT (VEC, elementwise-tile-tile, unary)
// Source: docs/tileop-usage/engines.md — VEC | TNOT | elementwise-tile-tile.
// NOTE(doc-gap): no signature in docs; bitwise NOT -> integer dtype inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], c[NE];
    gfill_idx(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<int32_t, M, N>(c, a, [](auto& d, auto& s){ TNOT(d, s); });
    BENCHEND;
    return 0;
}
