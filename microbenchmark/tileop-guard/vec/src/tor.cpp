#include "guard_common.hpp"
// TileOP-API doc guard: TOR (VEC, elementwise-tile-tile, bitwise binary)
// Source: docs/tileop-usage/engines.md — VEC | TOR | elementwise-tile-tile.
// NOTE(doc-gap): no signature/dtype in docs; bitwise → int32 chosen.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], b[NE], c[NE];
    for (int i = 0; i < NE; ++i) { a[i] = i; b[i] = 0x0000ff00; c[i] = 0; }
    BENCHSTART;
    g_binary<int32_t, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TOR(d, s0, s1); });
    BENCHEND;
    return 0;
}
