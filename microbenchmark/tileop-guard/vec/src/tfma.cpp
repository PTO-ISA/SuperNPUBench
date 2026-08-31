#include "guard_common.hpp"
// TileOP-API doc guard: TFMA (VEC, elementwise-tile-tile, fused multiply-add)
// Source: docs/tileop-usage/engines.md — VEC | TFMA | elementwise-tile-tile.
// NOTE(doc-gap): no signature/example in docs; ternary (dst,a,b,c) inferred
// from the FMA name + dst-first convention. Operand order unverified.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], b[NE], e[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE); gfill_seq(e, NE); gzero(c, NE);
    BENCHSTART;
    g_ternary<float, M, N>(c, a, b, e,
        [](auto& d, auto& s0, auto& s1, auto& s2){ TFMA(d, s0, s1, s2); });
    BENCHEND;
    return 0;
}
