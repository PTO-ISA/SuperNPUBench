#include "guard_common.hpp"
// TileOP-API doc guard: TADD (VEC, elementwise-tile-tile, binary)
// Source: docs/tileop-usage/engines.md — VEC | TADD | elementwise-tile-tile.
// NOTE(doc-gap): engines.md lists the op + class but gives NO signature for
// basic arithmetic; the (dst,src0,src1) call shape is inferred from the
// dst-first convention shown in cmp.md / cube.md.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], b[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE); gzero(c, NE);
    BENCHSTART;
    g_binary<float, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TADD(d, s0, s1); });
    BENCHEND;
    return 0;
}
