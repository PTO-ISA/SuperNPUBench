#include "guard_common.hpp"
// TileOP-API doc guard: TDIV (VEC, elementwise-tile-tile, binary)
// Source: docs/tileop-usage/engines.md — VEC | TDIV | elementwise-tile-tile.
// NOTE(doc-gap): no signature in docs; (dst,src0,src1) inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], b[NE], c[NE];
    gfill_seq(a, NE, 1.0f); gfill_seq(b, NE, 1.0f); gzero(c, NE);
    BENCHSTART;
    g_binary<float, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TDIV(d, s0, s1); });
    BENCHEND;
    return 0;
}
