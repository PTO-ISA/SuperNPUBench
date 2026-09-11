#include "guard_common.hpp"
// TileOP-API doc guard: TPREFETCH (TLSU) — prefetch GM range, no tile binding.
// Source: docs/tileop-usage/tlsu.md — FULL signature:
//   TPREFETCH(src, valid_col, valid_row)
// LB0/LB1 = valid shape, LB2 = physical row width, B.IOR = base + row stride.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE];
    gfill_seq(a, NE);
    // doc: src is a "static RowMajor global_tensor" (physical row width from
    // its compile-time column count) -> use gm_t directly, not an iterator view.
    gm_t<float, M, N> gA((float *)a);
    BENCHSTART;
    TPREFETCH(gA, N, M);   // valid_col=N, valid_row=M
    BENCHEND;
    return 0;
}
