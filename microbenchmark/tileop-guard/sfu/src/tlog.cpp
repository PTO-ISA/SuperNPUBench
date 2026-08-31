#include "guard_common.hpp"
// TileOP-API doc guard: TLOG (SFU, elementwise-tile-tile)
// Source: docs/tileop-usage/engines.md — SFU | TLOG | elementwise-tile-tile.
// NOTE(doc-gap): no signature in docs; (dst,src) unary shape inferred. log; src filled >=1 to stay in domain
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE, 1.0f); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TLOG(d, s); });
    BENCHEND;
    return 0;
}
