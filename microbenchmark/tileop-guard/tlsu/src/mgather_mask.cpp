#include "guard_common.hpp"
// TileOP-API doc guard: MGATHER_MASK (TLSU irregular) — masked gather.
// Source: docs/tileop-usage/tlsu.md "Gather and scatter": offset AND mask tiles
//   are Local operands. NO C++ signature given. Guessing
//   MGATHER_MASK(dst, base, offsetTile, maskTile, validCol, validRow).
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float d[NE];
    uint32_t off[NE], msk[NE];
    gzero(d, NE);
    for (int i = 0; i < NE; ++i) { off[i] = (uint32_t)((i * 4) % (NE * 4)); msk[i] = (i & 1); }

    iter_t<float, M, N> gD(d);
    iter_t<uint32_t, M, N> gO((uint32_t *)off), gMask((uint32_t *)msk);
    auto gD0 = gD(0, 0);
    auto gO0 = gO(0, 0);
    auto gM0 = gMask(0, 0);
    vtile_t<float, M, N> tD;
    vtile_t<uint32_t, M, N> tOff, tMask;
    TLOAD(tOff, gO0);
    TLOAD(tMask, gM0);
    BENCHSTART;
    MGATHER_MASK(tD, (uint64_t)0, tOff, tMask, N, M);
    BENCHEND;
    TSTORE(gD0, tD);
    return 0;
}
