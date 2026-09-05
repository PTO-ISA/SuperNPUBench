#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TSEL — masked select, IN-PLACE on dst.
// v0.58: TSEL(dst, mask, true_src) with dst[i] = (mask[i]) ? true_src[i] : dst_prior[i].
// The mask MUST be a packed predicate (logical tile) — gfrun rejects a plain
// data tile ("select first B.IOT requires mask then true/source Tile"). The only
// producer of a predicate is TCMP/TCMPS, so a correct TSEL demo chains them:
//   TCMP<LT>(mask, a, b);  TSEL(dst=prior, mask, tru)  => out = where(a<b, tru, prior)
// docs (engines.md) give NO signature/semantics; recovered from the header +
// gfrun contract. Precision: res_check.
constexpr int M = 16, N = 16, NE = M * N;
static int32_t A[NE], B[NE], prior[NE], tru[NE], out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", A, sizeof(A));
    guard_read_bin(CHK_DIR "/in_b.bin", B, sizeof(B));
    guard_read_bin(CHK_DIR "/in_c.bin", prior, sizeof(prior));
    guard_read_bin(CHK_DIR "/in_d.bin", tru, sizeof(tru));
#else
    for (int i = 0; i < NE; ++i) { A[i] = i - 8; B[i] = (i * 3) % 11 - 5; prior[i] = -i - 1; tru[i] = i + 100; }
#endif
    iter_t<int32_t, M, N> gA(A), gB(B), gP(prior), gT(tru), gO(out);
    auto a0 = gA(0, 0), b0 = gB(0, 0), p0 = gP(0, 0), t0 = gT(0, 0), o0 = gO(0, 0);
    vtile_t<int32_t, M, N> tA, tB, tMask, tD, tTru;
    TLOAD(tA, a0);
    TLOAD(tB, b0);
    TLOAD(tD, p0);         // dst prior = false source (in-place)
    TLOAD(tTru, t0);
    BENCHSTART;
    TCMP<CmpMode::LT>(tMask, tA, tB);   // predicate feeding TSEL
    TSEL(tD, tMask, tTru);              // dst = mask ? tru : prior
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
