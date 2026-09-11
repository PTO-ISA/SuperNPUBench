#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: THISTOGRAM — byte-indexed histogram.
// v0.58 signature: THISTOGRAM(dst, src, Idx, int ByteId).
//   Two source tiles (data src + index Idx) + a ByteId selector (0..3); output
//   defaults to UINT32 bins. docs/tileop-usage gives NO signature; 4-arg form
//   recovered from the header. Bin/accumulation semantics are not pinned by
//   docs, so this stays a run-only stability guard (no golden).
constexpr int M = 16, N = 16, NE = M * N;
static int32_t src[NE], idx[NE];
static uint32_t out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
    guard_read_bin(CHK_DIR "/in_b.bin", idx, sizeof(idx));
#else
    for (int i = 0; i < NE; ++i) { src[i] = (i * 7) % 251; idx[i] = i % N; }
#endif
    iter_t<int32_t, M, N> gS(src), gX(idx);
    iter_t<uint32_t, M, N> gO(out);
    auto gS0 = gS(0, 0);
    auto gX0 = gX(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<int32_t, M, N> tS, tX;
    vtile_t<uint32_t, M, N> tD;
    TLOAD(tS, gS0);
    TLOAD(tX, gX0);
    BENCHSTART;
    THISTOGRAM(tD, tS, tX, 0);
    BENCHEND;
    TSTORE(gO0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
