#include "guard_common.hpp"
// TileOP-API doc guard: TXORS (VEC, tile-scalar-and-immediate, integer)
// Source: docs/tileop-usage/engines.md — VEC | TXORS | tile-scalar-and-immediate.
// NOTE(doc-gap): no signature/dtype in docs; (dst,src,scalar) int32 inferred.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], c[NE];
    for (int i=0;i<NE;++i){ a[i]=i+1; c[i]=0; }
    BENCHSTART;
    g_scalar<int32_t, M, N>(c, a, 2, [](auto& d, auto& s, int32_t k){ TXORS(d, s, k); });
    BENCHEND;
    return 0;
}
