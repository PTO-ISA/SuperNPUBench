#include "guard_common.hpp"
// TileOP-API doc guard: TTRI (SFU irregular-and-complex).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature; no
// dedicated doc page.
// NOTE(doc-gap): triangular fill/mask presumably; NO signature documented.
// Guess: (dst, src) unary; recorded in REPORT as "no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TTRI(d, s); });
    BENCHEND;
    return 0;
}
