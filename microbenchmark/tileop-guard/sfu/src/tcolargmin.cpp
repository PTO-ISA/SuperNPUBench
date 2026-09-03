#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TCOLARGMIN — per-column argmin index. Output dtype forced
// to UINT32 (argReduce); dst physical MxN with ValidRow=1 (index at out[0*N+c]).
// See trowargmax.cpp for the full contract note. Precision: res_check.
constexpr int M = 16, N = 16, NE = M * N;
static float src[NE];
static uint32_t out[NE];
using OutTile = Tile<Location::Vec, uint32_t, M, N, BLayout::RowMajor, 1, N>;
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#else
    for (int i = 0; i < NE; ++i) src[i] = (float)((i * 37) % 101);
#endif
    iter_t<float, M, N> gS(src);
    global_iterator<gm_t<uint32_t, M, N>, OutTile> gO(out);
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
