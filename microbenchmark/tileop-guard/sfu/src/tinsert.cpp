#include "guard_common.hpp"
// TileOP-API doc guard: TINSERT (SFU layout-and-rearrangement).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature.
// NOTE(doc-gap): layout.md prose mentions insertion exists but gives NO
// signature/operand contract. Guess: (dst, src) binary insert; recorded in
// REPORT as "documentation gives no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], b[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE, 1.0f); gzero(c, NE);
    BENCHSTART;
    g_binary<float, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TINSERT(d, s0, s1); });
    BENCHEND;
    return 0;
}
