#include "guard_common.hpp"
// TileOP-API doc guard: TEXTRACT (SFU layout-and-rearrangement).
// Source: docs/tileop-usage/engines.md ONLY — one row, no signature.
// NOTE(doc-gap): layout.md prose mentions extraction exists but gives NO
// signature/operand contract. Guess: (dst, src) sub-tile extract; recorded
// in REPORT as "documentation gives no usable signature".
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TEXTRACT(d, s); });
    BENCHEND;
    return 0;
}
