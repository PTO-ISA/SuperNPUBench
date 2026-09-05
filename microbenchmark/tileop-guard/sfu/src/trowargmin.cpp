#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TROWARGMIN — per-row argmin index. Output dtype forced to
// UINT32 (argReduce). ops-20260904 TileOP-API requires a GENUINE single-column
// dst (ValidCol==1 && Cols==1); index at out[r] of an M x 1 tile.
// See trowargmax.cpp for the full contract note. Precision: res_check.
constexpr int M = 16, N = 16, NE = M * N;
static float src[NE];
static uint32_t out[M * 1];
using OutTile = Tile<Location::Vec, uint32_t, M, 1, BLayout::RowMajor, M, 1>;
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#else
    for (int i = 0; i < NE; ++i) src[i] = (float)((i * 37) % 101);
#endif
    iter_t<float, M, N> gS(src);
    global_iterator<gm_t<uint32_t, M, 1>, OutTile> gO(out);
    auto s0 = gS(0, 0);
    auto o0 = gO(0, 0);
    vtile_t<float, M, N> tS;
    OutTile tD;
    TLOAD(tS, s0);
    BENCHSTART;
    TROWARGMIN(tD, tS);
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
