#include "guard_common.hpp"
// TileOP-API doc guard: TPARTMIN (SFU irregular-and-complex).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature; no
// dedicated doc page.
// NOTE(doc-gap): partitioned/segmented min presumably; NO signature documented.
// Guess: (dst, src0, src1) binary; recorded in REPORT as "no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], b[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE, 1.0f); gzero(c, NE);
    BENCHSTART;
    g_binary<float, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TPARTMIN(d, s0, s1); });
    BENCHEND;
    return 0;
}
