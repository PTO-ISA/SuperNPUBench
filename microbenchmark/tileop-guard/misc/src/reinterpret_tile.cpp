#include "guard_common.hpp"
// TileOP-API doc guard: reinterpret_tile<NewDType>(src) (misc) — zero-instr bitcast view.
// Source: docs/tileop-usage/reinterpret-tile.md — FULL signature + examples.
//   auto v = reinterpret_tile<int32_t>(src);  // same bit width, Local only,
//   no TCVT/copy; view is non-owning, keep src alive.
// Contract: same bit width (fp32<->int32 OK), Local tile, registered dtype.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE];
    int32_t c[NE];
    gfill_seq(a, NE, 1.0f);
    for (int i = 0; i < NE; ++i) c[i] = 0;

    iter_t<float, M, N> gA((float *)a);
    iter_t<int32_t, M, N> gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA;
    vtile_t<int32_t, M, N> tC;
    TLOAD(tA, gA0);
    BENCHSTART;
    auto tA_s32 = reinterpret_tile<int32_t>(tA);   // fp32 bits viewed as int32
    TABS(tC, tA_s32);                              // consume the view (emits lb2)
    BENCHEND;
    TSTORE(gC0, tC);
    return 0;
}
