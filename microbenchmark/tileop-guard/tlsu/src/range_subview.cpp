#include "guard_common.hpp"
// TileOP-API doc guard: range::Subview — source-side range carrier over TSTORE.
// Source: docs/tileop-usage/range-modifiers.md — full signature + example.
//   template<Parent, SubviewSizeCode, Offset=0, RegSrc=2> class Subview;
//   auto sv = range::Subview<Src, 1, /*Off*/0, /*RegSrc*/0>(s, base);
//   TSTORE(gm, sv);  // emits B.SUBVIEW 0, r0, 0, 1 after the source B.IOT
//   SubviewSizeCode 1..12 (128 B..256 KiB); 4x8 float tile = 128 B -> code 1.
int main() {
    constexpr int M = 4, N = 8;
    float ha[M * N], hc[M * N];
    gfill_seq(ha, M * N);
    gzero(hc, M * N);

    using Src = vtile_t<float, M, N>;
    Src s;

    iter_t<float, M, N> gA(ha);
    auto gA0 = gA(0, 0);
    global_tensor<float, RowMajor<M, N>> gm(hc);

    TLOAD(s, gA0);
    BENCHSTART;
    auto sv = range::Subview<Src, 1, /*Off*/ 0, /*RegSrc*/ 0>(s, 0);
    TSTORE(gm, sv);   // B.SUBVIEW 0, r0, 0, 1
    BENCHEND;
    return 0;
}
