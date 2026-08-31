#include "guard_common.hpp"
// TileOP-API doc guard: MGATHER_CAS (TLSU irregular) — atomic compare-and-swap.
// Source: docs/tileop-usage/tlsu.md — FULL signature given.
//   template <DstTile, IndexTile, ExpectedTile, ReplacementTile>
//   void MGATHER_CAS(DstTile &observedOld, uint64_t base,
//                    IndexTile &byteDisplacements, ExpectedTile &expected,
//                    ReplacementTile &replacement,
//                    uint32_t validCol, uint32_t validRow = 1);
// Contract: expected/replacement/observedOld share one transfer dtype;
//   byteDisplacements is integer byte-displacement tile; all match ValidRow x ValidCol.
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float oldv[NE], exp[NE], rep[NE];
    uint32_t off[NE];
    gzero(oldv, NE); gfill_const(exp, NE, 0.0f); gfill_seq(rep, NE, 1.0f);
    for (int i = 0; i < NE; ++i) off[i] = (uint32_t)(i * 4);

    iter_t<float, M, N> gO(oldv), gE((float *)exp), gR((float *)rep);
    iter_t<uint32_t, M, N> gI((uint32_t *)off);
    auto gO0 = gO(0, 0);
    auto gE0 = gE(0, 0);
    auto gR0 = gR(0, 0);
    auto gI0 = gI(0, 0);
    vtile_t<float, M, N> tOld, tExp, tRep;
    vtile_t<uint32_t, M, N> tIdx;
    TLOAD(tExp, gE0);
    TLOAD(tRep, gR0);
    TLOAD(tIdx, gI0);
    BENCHSTART;
    MGATHER_CAS(tOld, /*base=*/0x1000, tIdx, tExp, tRep, N, M);
    BENCHEND;
    TSTORE(gO0, tOld);
    return 0;
}
