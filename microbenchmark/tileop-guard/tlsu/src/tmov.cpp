#include "guard_common.hpp"
// TileOP-API doc guard: TMOV (TLSU) — tile-to-tile move (Local).
// Source: docs/tileop-usage/tlsu.md + engines.md (function 2). No explicit C++
// signature for the base Local form; `(dst, src)` inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    iter_t<float, M, N> gA((float *)a), gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA, tC;
    TLOAD(tA, gA0);
    BENCHSTART;
    TMOV(tC, tA);
    BENCHEND;
    TSTORE(gC0, tC);
    return 0;
}
