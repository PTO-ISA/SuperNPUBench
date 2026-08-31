#include "guard_common.hpp"
// TileOP-API doc guard: TEXPANDS (VEC, tile-scalar-and-immediate)
// Source: docs/tileop-usage/engines.md — VEC | TEXPANDS | tile-scalar-and-immediate.
// NOTE(doc-gap): docs give NO signature. The real API is 2-arg
// TEXPANDS(dst, scalar) — it fills a tile with a broadcast scalar and takes
// NO source tile (compiler feedback). Recorded in REPORT.md.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float c[NE];
    gzero(c, NE);
    iter_t<float, M, N> gC(c);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tC;
    BENCHSTART;
    TEXPANDS(tC, 1.5f);
    TSTORE(gC0, tC);
    BENCHEND;
    return 0;
}
