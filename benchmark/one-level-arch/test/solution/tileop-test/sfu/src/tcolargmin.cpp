#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TCOLARGMIN — per-column argmin index. Output dtype forced
// to UINT32 (argReduce). ops-20260904 model writes col reductions into a
// GENUINE single-row 1 x N tile (index at out[c]); the historical physical
// M x N + ValidRow=1 workaround read stale bytes. See reducemax_colvec.hpp.
constexpr int M = 16, N = 16, NE = M * N;
static float src[NE];
static uint32_t out[1 * N];
using OutTile = Tile<Location::Vec, uint32_t, 1, N, BLayout::RowMajor>;
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#else
    for (int i = 0; i < NE; ++i) src[i] = (float)((i * 37) % 101);
#endif
    iter_t<float, M, N> gS(src);
    global_iterator<gm_t<uint32_t, 1, N>, OutTile> gO(out);
    auto s0 = gS(0, 0);
    auto o0 = gO(0, 0);
    vtile_t<float, M, N> tS;
    OutTile tD;
    TLOAD(tS, s0);
    BENCHSTART;
    TCOLARGMIN(tD, tS);
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
