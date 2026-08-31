#include "guard_common.hpp"
// TileOP-API doc guard: TCMP (VEC, elementwise-tile-tile, compare)
// Source: docs/tileop-usage/cmp.md — TCMP<Mode>(dst, src0, src1).
// Doc gives a full signature + example. Using GT (from the doc example).
// DType table in cmp.md: int32 supports all 6 modes -> use int32 in/out.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], b[NE], c[NE];
    gfill_seq(a, NE); gfill_seq(b, NE, (int32_t)3); gzero(c, NE);
    BENCHSTART;
    g_binary<int32_t, M, N>(c, a, b,
        [](auto& d, auto& s0, auto& s1){ TCMP<CmpMode::GT>(d, s0, s1); });
    BENCHEND;
    return 0;
}
