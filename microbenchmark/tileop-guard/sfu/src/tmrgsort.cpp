#include "guard_common.hpp"
// TileOP-API doc guard: TMRGSORT (SFU irregular) — merge two sorted rows.
// Source: docs/tileop-usage/sort.md — FULL signature given.
//   template <DstTile, LeftTile, RightTile>
//   void TMRGSORT(dst, left, right, descending=false);
// Contract: dst/left/right share one FP16/FP32 dtype; sources are persistent
//   sorted single-row Local tiles.
// NOTE(doc-example-bug): sort.md example uses `Row a,b,out` ALL at 1x256, but a
// static_assert requires dst.ValidCol == left.ValidCol + right.ValidCol (with
// dst.Cols a power of two and Cols/2 < combined <= Cols). The doc example does
// NOT compile. Correct shapes: two 1x128 sources merged into a 1x256 dst.
int main() {
    constexpr int H = 128, W = 256;
    float a[H], b[H], out[W];
    // two ascending-sorted single-row sources
    for (int i = 0; i < H; ++i) { a[i] = (float)(2 * i); b[i] = (float)(2 * i + 1); }
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
    return 0;
}
