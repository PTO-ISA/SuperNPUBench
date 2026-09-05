#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TSCATTER (SFU irregular-and-complex, layout).
// Authoritative semantics (pto-spec normative ASL, irregular-and-complex/layout):
//   "For source coordinate [r,c], index value k writes the source bits to
//   destination [k,c]; duplicate destination coordinates are illegal."
//   -> dst[idx[r,c], c] = src[r,c]  (index picks the ROW; column stays; index must
//   be injective within each column). Index dtype ∈ {S16,U16,S32,U32,S64,U64};
//   signature TSCATTER(dst, src, index) matches engines.md 3-operand form.
// Precision: res_check READY golden — TSCATTER currently run-fails (model gap);
//   when the model implements it, check_tscatter auto-validates.
constexpr int M = 16, N = 16, NE = M * N;
static float a[NE], c[NE];
static int32_t idx[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", a, sizeof(a));
    guard_read_bin(CHK_DIR "/in_idx.bin", idx, sizeof(idx));
#else
    gfill_seq(a, NE); gfill_idx(idx, NE);
#endif
    gzero(c, NE);
    iter_t<float, M, N> gA((float *)a), gC(c);
    iter_t<int32_t, M, N> gI(idx);
    auto gA0 = gA(0, 0);
    auto gI0 = gI(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA, tC;
    vtile_t<int32_t, M, N> tI;
    TLOAD(tA, gA0);
    TLOAD(tI, gI0);
    BENCHSTART;
    TSCATTER(tC, tA, tI);
    BENCHEND;
    TSTORE(gC0, tC);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", c, sizeof(c));
#endif
    return 0;
}
