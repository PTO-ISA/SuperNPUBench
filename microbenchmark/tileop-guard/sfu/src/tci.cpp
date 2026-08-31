#include "guard_common.hpp"
// TileOP-API doc guard: TCI (SFU irregular) — contiguous integer sequence.
// Source: docs/tileop-usage/tci.md — FULL signature + operand contract given.
//   template <is_tile_data_v Tile, typename T, int descending=0>
//   void TCI(Tile &dst, T start);
// Contract: dst = ordinary Local VEC RowMajor tile; dtype exactly S32/S16/U32/
//   U16; start same C++ type as element; ValidRow == 1.
int main() {
    constexpr int N = 64;
    int32_t out[N];
    gzero(out, N);
    iter_t<int32_t, 1, N> gO(out);
    auto gO0 = gO(0, 0);
    vtile_t<int32_t, 1, N> t;
    BENCHSTART;
    TCI(t, static_cast<int32_t>(0));   // ascending: col k -> start + k
    BENCHEND;
    TSTORE(gO0, t);
    return 0;
}
