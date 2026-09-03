#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TSELS — masked select between a tile source and a scalar.
// v0.58: TSELS(dst, mask, scalar, src1). gfrun contract: srcs[1] is a logical
// predicate (mask), srcs[2] a data source; dst is a fresh output (not in-place).
// So the mask must come from TCMP. End-to-end:
//   TCMP<GT>(mask, a, b);  TSELS(dst, mask, SVAL, src)  => out = where(a>b, src, SVAL)
// (polarity confirmed by the res_check golden). docs give NO signature; the
// scalar-in-the-middle 4-arg form + predicate contract are recovered from the
// header + gfrun. Precision: res_check.
constexpr int M = 16, N = 16, NE = M * N;
constexpr int32_t SVAL = 777;
static int32_t A[NE], B[NE], src[NE], out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", A, sizeof(A));
    guard_read_bin(CHK_DIR "/in_b.bin", B, sizeof(B));
    guard_read_bin(CHK_DIR "/in_c.bin", src, sizeof(src));
#else
    for (int i = 0; i < NE; ++i) { A[i] = i - 8; B[i] = (i * 3) % 11 - 5; src[i] = i + 100; }
#endif
    iter_t<int32_t, M, N> gA(A), gB(B), gS(src), gO(out);
    auto a0 = gA(0, 0), b0 = gB(0, 0), s0 = gS(0, 0), o0 = gO(0, 0);
    vtile_t<int32_t, M, N> tA, tB, tMask, tSrc, tD;
    TLOAD(tA, a0);
    TLOAD(tB, b0);
    TLOAD(tSrc, s0);
    BENCHSTART;
    TCMP<CmpMode::GT>(tMask, tA, tB);       // predicate: a > b
    TSELS(tD, tMask, SVAL, tSrc);           // dst = mask ? src : SVAL
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
