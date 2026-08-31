#include "guard_common.hpp"
// TileOP-API doc guard: MSCATTER (TLSU irregular) — scatter to GM by offsets.
// Source: docs/tileop-usage/tlsu.md "Gather and scatter": offset/mask tiles are
//   Local operands; base + row stride scalar. NO C++ signature given.
//   Guessing MSCATTER(base, offsetTile, srcTile, validCol, validRow).
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float s[NE];
    uint32_t off[NE];
    gfill_seq(s, NE);
    for (int i = 0; i < NE; ++i) off[i] = (uint32_t)((i * 4) % (NE * 4));

    iter_t<float, M, N> gS((float *)s);
    iter_t<uint32_t, M, N> gO((uint32_t *)off);
    auto gS0 = gS(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<float, M, N> tS;
    vtile_t<uint32_t, M, N> tOff;
    TLOAD(tS, gS0);
    TLOAD(tOff, gO0);
    BENCHSTART;
    MSCATTER((uint64_t)0, tOff, tS, N, M);
    BENCHEND;
    return 0;
}
