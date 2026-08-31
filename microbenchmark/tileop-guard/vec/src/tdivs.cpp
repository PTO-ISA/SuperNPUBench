#include "guard_common.hpp"
// TileOP-API doc guard: TDIVS (VEC, tile-scalar-and-immediate)
// Source: docs/tileop-usage/engines.md — VEC | TDIVS | tile-scalar-and-immediate.
// NOTE(doc-gap): no signature for tile-scalar ops in docs; (dst,src,scalar)
// inferred from the class name + TCMPS signature (cmp.md) which travels the
// scalar via B.IOR.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_scalar<float, M, N>(c, a, 2.0f, [](auto& d, auto& s, float k){ TDIVS(d, s, k); });
    BENCHEND;
    return 0;
}
