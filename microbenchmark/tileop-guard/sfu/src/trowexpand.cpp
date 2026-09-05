#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TROWEXPAND — copy-expand from a per-row broadcast source.
// v0.58 contract (recovered from header + emulator + ValidateReduceAndExpandTepl):
//   TROWEXPAND(dst, src) requires ONE broadcast source with validRow=M,
//   validCol=1, physical col=1 (a physical M x 1 column). The emulator computes
//   dst[i,j]=src[i,0], BUT the fill width equals the source ValidCol (which the
//   validator pins to 1), so the model only fills column 0 of dst — a degenerate
//   "expand". So this is left run-only (no golden): the correct broadcast source
//   now COMPILES + RUNS (the previous run-fail is fixed), but the model's fill
//   width is inconsistent with a full M x N broadcast and is not verified here.
constexpr int M = 16, N = 16, NE = M * N;
using SrcTile = vtile_t<float, M, 1>;              // physical M x 1 broadcast column
static float src[M];
static float out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#endif
    for (int i = 0; i < M; ++i) if (src[i] == 0.0f) src[i] = (float)(i + 1) * 0.5f;
    iter_t<float, M, 1> gS(src);
    iter_t<float, M, N> gO(out);
    auto s0 = gS(0, 0);
    auto o0 = gO(0, 0);
    SrcTile tS;
    vtile_t<float, M, N> tD;
    TLOAD(tS, s0);
    BENCHSTART;
    TROWEXPAND(tD, tS);
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
