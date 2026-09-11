#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TCI (SFU irregular/initialization) — contiguous integer
// sequence generation ("create index" / vci).
// Source: docs/tileop-usage/irregular-and-complex/initialization/TCI.md — FULL
//   signature + example given:
//     template <is_tile_data_v tile_shape, typename T, int descending=0>
//     void TCI(tile_shape &dst, T s);
//   Example: TileT = Tile<Vec,int32_t,1,64>; TCI<TileT,int32_t,0>(dst, 0); TSTORE.
//   Semantics (doc prose): one row, ValidRow==1; ascending col k = start+k,
//   descending = start-k; dtype in {S32,S16,U32,U16}.
// Precision: res_check. TCI takes NO input tile (self-generates), so golden.py
//   gen is a no-op; the numpy oracle recomputes the iota independently.
constexpr int GN = 64;                 // 1x64 int32 == 256 B (doc example shape)
static int32_t out[GN];
int main() {
    using TileT = vtile_t<int32_t, 1, GN>;
    iter_t<int32_t, 1, GN> gO(out);
    auto gO0 = gO(0, 0);
    TileT t;
    BENCHSTART;
    TCI<TileT, int32_t, 0>(t, 0);      // ascending from 0: col k -> k
    BENCHEND;
    TSTORE(gO0, t);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
