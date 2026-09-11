#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TCI descending variant (create index / vci).
// Source: TCI.md — the third template parameter `descending` selects direction;
//   descending col k = start - k. Here start=100 -> 100,99,98,...
constexpr int GN = 64;
static int32_t out[GN];
int main() {
    using TileT = vtile_t<int32_t, 1, GN>;
    iter_t<int32_t, 1, GN> gO(out);
    auto gO0 = gO(0, 0);
    TileT t;
    BENCHSTART;
    TCI<TileT, int32_t, 1>(t, 100);    // descending from 100: col k -> 100-k
    BENCHEND;
    TSTORE(gO0, t);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
