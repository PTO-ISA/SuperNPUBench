#include "guard_common.hpp"
// TileOP-API doc guard: TIMG2COL (SFU layout) — image-to-column transform.
// Source: docs/tileop-usage/quant-and-im2col.md — FULL signature + example.
//   template <...out, ...in> void TIMG2COL(out &dst, in &src,
//                                          uint32_t posM=0, uint32_t posK=0);
// NOTE(doc): docs say src's feature-map descriptor (NC1HWC0, filter, stride...)
// is a property of a persistent Matrix-location source tile, but the doc EXAMPLE
// uses a plain Location::Vec RowMajor tile. Following the example verbatim.
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){
        TIMG2COL(d, s, /*posM=*/3, /*posK=*/5);
    });
    BENCHEND;
    return 0;
}
