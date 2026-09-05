#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMRGSORT (SFU irregular) — merge two sorted rows.
// Source: docs/tileop-usage/sort.md — FULL signature given.
//   template <DstTile, LeftTile, RightTile>
//   void TMRGSORT(dst, left, right, descending=false);
// Authoritative semantics (pto-spec normative ASL, irregular-and-complex/sorting):
//   "stably merge two sorted single-row Local streams into one destination".
//   -> dst = ascending-merge(left, right) = sorted(concat(left,right)) for sorted
//   inputs. NOTE(doc-example-bug): sort.md example uses 1x256 for all three, but a
//   static_assert requires dst.ValidCol == left.ValidCol + right.ValidCol; correct
//   shapes are two 1x128 sources merged into a 1x256 dst.
// Precision: res_check READY golden — TMRGSORT is currently an unimplemented model
// stub (run-fail); when the model implements it, check_mrgsort auto-validates.
constexpr int H = 128, W = 256;
static float a[H], b[H], out[W];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", a, sizeof(a));   // ascending-sorted left
    guard_read_bin(CHK_DIR "/in_b.bin", b, sizeof(b));   // ascending-sorted right
#else
    for (int i = 0; i < H; ++i) { a[i] = (float)(2 * i); b[i] = (float)(2 * i + 1); }
#endif
    gzero(out, W);
    iter_t<float, 1, H> gA((float *)a), gB((float *)b);
    iter_t<float, 1, W> gO(out);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gO0 = gO(0, 0);
    vtile_t<float, 1, H> tA, tB;
    vtile_t<float, 1, W> tO;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    BENCHSTART;
    TMRGSORT(tO, tA, tB);              // ascending merge
    BENCHEND;
    TSTORE(gO0, tO);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
