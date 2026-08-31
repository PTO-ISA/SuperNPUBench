#include "guard_common.hpp"
// TileOP-API doc guard: TSTORE (TLSU) — Local tile -> GM.
// Source: docs/tileop-usage/tlsu.md — TSTORE(iterator, tile).
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    iter_t<float, M, N> gA((float *)a), gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> t;
    TLOAD(t, gA0);
    BENCHSTART;
    TSTORE(gC0, t);
    BENCHEND;
    return 0;
}
