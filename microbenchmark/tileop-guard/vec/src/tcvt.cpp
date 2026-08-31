#include "guard_common.hpp"
// TileOP-API doc guard: TCVT (VEC, elementwise-tile-tile, numeric conversion)
// Source: docs/tileop-usage/reinterpret-tile.md — contrasts TCVT (numeric
// conversion, emits hardware op) vs reinterpret_tile (bit view). Signature
// shown there: TCVT(converted, src) with distinct dst dtype.
// NOTE(doc-gap): only the fp32->int32 form is exemplified; call shape reused.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE]; int32_t c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary_cvt<float, int32_t, M, N>(c, a, [](auto& d, auto& s){ TCVT(d, s); });
    BENCHEND;
    return 0;
}
