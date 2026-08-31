#include "guard_common.hpp"
// TileOP-API doc guard: TCMPS (VEC, tile-scalar compare)
// Source: docs/tileop-usage/cmp.md — TCMPS<Mode>(dst, src, scalar).
// Doc example: TCMPS<CmpMode::GE>(d, a, 0.5f). int32 supports all 6 modes.
int main() {
    constexpr int M = 16, N = 16, NE = M * N;
    int32_t a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_scalar<int32_t, M, N>(c, a, (int32_t)5,
        [](auto& d, auto& s, auto sc){ TCMPS<CmpMode::GE>(d, s, sc); });
    BENCHEND;
    return 0;
}
