#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TCMPS — tile-vs-scalar compare producing a packed predicate.
// v0.58: TCMPS<Mode>(dst, src, scalar). Like TCMP, the result is a predicate
// that cannot be TSTORE'd directly; it is consumed by TSEL. End-to-end usage:
//   TCMPS<GT>(mask, a, 0);  TSEL(dst=prior, mask, tru)  => out = where(a>0, tru, prior)
// cmp.md documents the signature but not the predicate-consumption contract.
// Precision: res_check.
constexpr int M = 16, N = 16, NE = M * N;
static int32_t A[NE], prior[NE], tru[NE], out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", A, sizeof(A));
    guard_read_bin(CHK_DIR "/in_b.bin", prior, sizeof(prior));
    guard_read_bin(CHK_DIR "/in_c.bin", tru, sizeof(tru));
#else
    for (int i = 0; i < NE; ++i) { A[i] = i - 8; prior[i] = -i - 1; tru[i] = i + 100; }
#endif
    iter_t<int32_t, M, N> gA(A), gP(prior), gT(tru), gO(out);
    auto a0 = gA(0, 0), p0 = gP(0, 0), t0 = gT(0, 0), o0 = gO(0, 0);
    vtile_t<int32_t, M, N> tA, tMask, tD, tTru;
    TLOAD(tA, a0);
    TLOAD(tD, p0);         // dst prior = false source (in-place)
    TLOAD(tTru, t0);
    BENCHSTART;
    TCMPS<CmpMode::GT>(tMask, tA, 0);   // predicate: a > 0
    TSEL(tD, tMask, tTru);              // dst = (a>0) ? tru : prior
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
