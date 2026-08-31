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
    auto as = range::Assemble<Dst, 12, /*INIT*/ true, /*LAST*/ false,
                              /*Off*/ 0, /*RegSrc*/ 0>(d, 0);
    TLOAD(as, gm);   // B.ASSEMBLE 1, 0, r0, 0, 12
    BENCHEND;
    TSTORE(gC0, d);
    return 0;
}
