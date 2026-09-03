#include "guard_common.hpp"
// TileOP-API doc guard: range::Assemble — destination-side range carrier over TLOAD.
// Source: docs/tileop-usage/range-modifiers.md — full signature + example.
//   template<Parent, ParentSizeCode, INIT=true, LAST=false, Offset=0, RegSrc=2>
//   auto as = range::Assemble<Dst, 12, /*INIT*/true, /*LAST*/false, 0, /*RegSrc*/0>(d, base);
//   TLOAD(as, gm);  // B.IOT ..., ->d / B.ASSEMBLE 1, 0, r0, 0, 12
//   INIT=1 requires ParentSizeCode 1..12.
int main() {
    constexpr int M = 4, N = 8;
    float ha[M * N], hc[M * N];
    gfill_seq(ha, M * N);
    gzero(hc, M * N);

    using Dst = vtile_t<float, M, N>;
    Dst d;

    global_tensor<float, RowMajor<M, N>> gm(ha);
    iter_t<float, M, N> gC(hc);
    auto gC0 = gC(0, 0);

    BENCHSTART;
    // ParentSizeCode must match the parent tile capacity: a 4x8 float tile is
    // 128 B = SizeCode 1 (the doc example's 12 = 256 KiB over-runs the tile and
    // trips a static_assert "B.ASSEMBLE length cannot exceed the parent Tile
    // capacity"). See range-modifiers.md docs issue.
    auto as = range::Assemble<Dst, 1, /*INIT*/ true, /*LAST*/ false,
                              /*Off*/ 0, /*RegSrc*/ 0>(d, 0);
    TLOAD(as, gm);   // B.ASSEMBLE 1, 0, r0, 0, 1
    BENCHEND;
    TSTORE(gC0, d);
    return 0;
}
