#include "guard_common.hpp"
// TileOP-API doc guard: MGATHER (TLSU irregular) — gather from GM by offsets.
// Source: docs/tileop-usage/tlsu.md "Gather and scatter": offset/mask tiles are
//   Local operands; global base + row stride are scalar inputs. NO C++ signature
//   is given. Guessing MGATHER(dst, base, offsetTile, validCol, validRow) after
//   the MGATHER_CAS signature shape.
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float d[NE];
    uint32_t off[NE];
    gzero(d, NE);
    for (int i = 0; i < NE; ++i) off[i] = (uint32_t)((i * 4) % (NE * 4));

    iter_t<float, M, N> gD(d);
    iter_t<uint32_t, M, N> gO((uint32_t *)off);
    auto gD0 = gD(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<float, M, N> tD;
    vtile_t<uint32_t, M, N> tOff;
    TLOAD(tOff, gO0);
    BENCHSTART;
    MGATHER(tD, (uint64_t)0, tOff, N, M);
    BENCHEND;
    TSTORE(gD0, tD);
    return 0;
}
