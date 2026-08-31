#include "guard_common.hpp"
// TileOP-API doc guard: THISTOGRAM (SFU irregular-and-complex).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature; no
// dedicated doc page.
// NOTE(doc-gap): histogram bin-count presumably; NO signature documented.
// Guess: (dst, src) unary; recorded in REPORT as "no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], c[NE];
    gfill_idx(a, NE); for (int i = 0; i < NE; ++i) c[i] = 0;
    BENCHSTART;
    g_unary<int32_t, M, N>(c, a, [](auto& d, auto& s){ THISTOGRAM(d, s); });
    BENCHEND;
    return 0;
}
