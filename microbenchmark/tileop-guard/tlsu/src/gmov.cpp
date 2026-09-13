#include "guard_common.hpp"
// TileOP-API doc guard: GMOV (TLSU peer movement) — move tile between PEs.
// Source: docs/tileop-usage/tlsu.md — example given:  GMOV<15>(dst, peer_tid, src);
// Contract: all four PEs must reach the same dynamic instance; PEMask selects
//   requesters only. Peer TID carried by B.IOR.
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    iter_t<float, M, N> gA((float *)a), gC(c);
    auto gA0 = gA(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA, tC;
    TLOAD(tA, gA0);
    BENCHSTART;
    GMOV<15>(tC, /*peer_tid=*/0, tA);
    BENCHEND;
    TSTORE(gC0, tC);
    return 0;
}
