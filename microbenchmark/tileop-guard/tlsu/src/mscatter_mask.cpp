#include "guard_common.hpp"
// TileOP-API doc guard: MSCATTER_MASK (TLSU irregular) — masked scatter.
// Source: docs/tileop-usage/tlsu.md "Gather and scatter": offset AND mask tiles
//   are Local operands. NO C++ signature given. Guessing
//   MSCATTER_MASK(base, offsetTile, maskTile, srcTile, validCol, validRow).
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float s[NE];
    uint32_t off[NE], msk[NE];
    gfill_seq(s, NE);
    for (int i = 0; i < NE; ++i) { off[i] = (uint32_t)((i * 4) % (NE * 4)); msk[i] = (i & 1); }

    iter_t<float, M, N> gS((float *)s);
    iter_t<uint32_t, M, N> gO((uint32_t *)off), gMask((uint32_t *)msk);
    auto gS0 = gS(0, 0);
    auto gO0 = gO(0, 0);
    auto gM0 = gMask(0, 0);
    vtile_t<float, M, N> tS;
    vtile_t<uint32_t, M, N> tOff, tMask;
    TLOAD(tS, gS0);
    TLOAD(tOff, gO0);
    TLOAD(tMask, gM0);
    BENCHSTART;
    MSCATTER_MASK((uint64_t)0, tOff, tMask, tS, N, M);
    BENCHEND;
    return 0;
}
