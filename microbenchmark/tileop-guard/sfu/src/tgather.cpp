#include "guard_common.hpp"
// TileOP-API doc guard: TGATHER (SFU irregular-and-complex).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature; no
// dedicated doc page.
// NOTE(doc-gap): in-tile gather by index presumably; NO signature documented.
// Guess: (dst, src, index) ternary; recorded in REPORT as "no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    int32_t idx[NE];
    gfill_seq(a, NE); gfill_idx(idx, NE); gzero(c, NE);
    // guess: value src + index src -> gathered dst
    iter_t<float, M, N> gA((float *)a), gC(c);
    iter_t<int32_t, M, N> gI(idx);
    auto gA0 = gA(0, 0);
    auto gI0 = gI(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA, tC;
    vtile_t<int32_t, M, N> tI;
    TLOAD(tA, gA0);
    TLOAD(tI, gI0);
    BENCHSTART;
    TGATHER(tC, tA, tI);
    BENCHEND;
    TSTORE(gC0, tC);
    return 0;
}
