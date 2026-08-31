#include "guard_common.hpp"
// TileOP-API doc guard: TSELS (VEC, tile-scalar-and-immediate)
// Source: docs/tileop-usage/engines.md — VEC | TSELS | tile-scalar-and-immediate.
// NOTE(doc-gap): docs give NO signature/semantics/dtype. Real API (compiler
// feedback) is 4-arg TSELS(dst, src0, scalar, src1) with the scalar in the
// MIDDLE — unusual and undocumented. int32 dtype mirrors TSEL. Recorded in
// REPORT.md.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], b[NE], c[NE];
    for (int i = 0; i < NE; ++i) { a[i] = i - 8; b[i] = i; c[i] = 0; }

    iter_t<int32_t, M, N> gA(a), gB(b), gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<int32_t, M, N> tA, tB, tC;

    BENCHSTART;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TSELS(tC, tA, 0, tB);
    TSTORE(gC0, tC);
    BENCHEND;
    return 0;
}
