#include "guard_common.hpp"
// TileOP-API doc guard: TSHL (VEC, elementwise-tile-tile, shift-left binary)
// Source: docs/tileop-usage/engines.md — VEC | TSHL | elementwise-tile-tile.
// NOTE(doc-gap): no signature/dtype in docs; shift → int32 chosen.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], b[NE], c[NE];
    for (int i = 0; i < NE; ++i) { a[i] = i + 1; b[i] = i % 5; c[i] = 0; }
    BENCHSTART;
    g_binary<int32_t, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TSHL(d, s0, s1); });
    BENCHEND;
    return 0;
}
