#include "guard_common.hpp"
// TileOP-API doc guard: TCOLEXPANDMAX (SFU, reduce-and-expand)
// Source: docs/tileop-usage/engines.md — SFU | TCOLEXPANDMAX | reduce-and-expand.
// NOTE(doc-gap): docs give NO signature. Name implies a col-reduce broadcast
// combined arithmetically with a second full tile; binary (dst,src0,src1)
// inferred after the unary form failed to compile (compiler feedback).
int main() {
    constexpr int M = 16, N = 16, NE = M*N;
    float a[NE], b[NE], c[NE];
    gfill_seq(a, NE, 1.0f); gfill_seq(b, NE, 2.0f); gzero(c, NE);
    BENCHSTART;
    g_binary<float, M, N>(c, a, b, [](auto& d, auto& s0, auto& s1){ TCOLEXPANDMAX(d, s0, s1); });
    BENCHEND;
    return 0;
}
