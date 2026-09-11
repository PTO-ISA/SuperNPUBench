#include "guard_common.hpp"
// TileOP-API doc guard: TLOAD (TLSU) — GM -> Local tile.
// Source: docs/tileop-usage/tlsu.md — block layout given; C++ form is
//   TLOAD(tile, iterator). global_iterator supplies base + row stride.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    iter_t<float, M, N> gA((float *)a), gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> t;
    BENCHSTART;
    TLOAD(t, gA0);
    BENCHEND;
    TSTORE(gC0, t);
    return 0;
}
