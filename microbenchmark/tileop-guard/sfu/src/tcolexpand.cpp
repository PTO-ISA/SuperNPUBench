#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TCOLEXPAND — copy-expand from a per-column broadcast source.
// v0.58 contract (recovered from header + emulator + ValidateReduceAndExpandTepl):
//   TCOLEXPAND(dst, src) requires ONE broadcast source with validRow=1,
//   validCol=N (a 1 x N row, physical M x N tile with ValidRow=1). The emulator
//   computes dst[i,j]=src[0,j], BUT the fill height equals the source ValidRow
//   (pinned to 1), so the model only fills row 0 of dst — a degenerate "expand".
//   Left run-only (no golden): the correct broadcast source now COMPILES + RUNS
//   (previous run-fail fixed), but the degenerate fill height is not verified.
constexpr int M = 16, N = 16, NE = M * N;
using SrcTile = Tile<Location::Vec, float, M, N, BLayout::RowMajor, 1, N>;  // 1 x N broadcast row
static float src[NE];
static float out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#endif
    for (int i = 0; i < N; ++i) if (src[i] == 0.0f) src[i] = (float)(i + 1) * 0.25f;
    global_iterator<gm_t<float, M, N>, SrcTile> gS(src);
    iter_t<float, M, N> gO(out);
    auto s0 = gS(0, 0);
    auto o0 = gO(0, 0);
    SrcTile tS;
    vtile_t<float, M, N> tD;
    TLOAD(tS, s0);
    BENCHSTART;
    TCOLEXPAND(tD, tS);
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
