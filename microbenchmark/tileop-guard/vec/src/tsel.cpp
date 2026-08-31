#include "guard_common.hpp"
// TileOP-API doc guard: TSEL (VEC, elementwise-tile-tile)
// Source: docs/tileop-usage/engines.md — VEC | TSEL | elementwise-tile-tile.
// NOTE(doc-gap): engines.md lists TSEL as elementwise-tile-tile but gives NO
// signature and NO semantics. A 4-arg select(dst,cond,a,b) does not compile;
// the accepted form is 3-arg TSEL(dst, src0, src1) (compiler feedback). fp32
// is rejected, int32 accepted at compile. gfrun then asserts on the source
// tuple -> flagged as SUSPECTED ENV issue in REPORT.md (left as-is for review).
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], b[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE, (int32_t)3); gzero(c, NE);
    BENCHSTART;
    g_binary<int32_t, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TSEL(d, s0, s1); });
    BENCHEND;
    return 0;
}
