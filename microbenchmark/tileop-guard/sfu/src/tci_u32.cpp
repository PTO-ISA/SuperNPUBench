#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TCI U32 dtype (create index / vci).
// Source: TCI.md — supported dtypes S32/S16/U32/U16. Ascending from 0.
constexpr int GN = 64;                 // 1x64 u32 == 256 B
static uint32_t out[GN];
int main() {
    using TileT = vtile_t<uint32_t, 1, GN>;
    iter_t<uint32_t, 1, GN> gO(out);
    auto gO0 = gO(0, 0);
    TileT t;
    BENCHSTART;
    TCI<TileT, uint32_t, 0>(t, 0u);
    BENCHEND;
    TSTORE(gO0, t);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
