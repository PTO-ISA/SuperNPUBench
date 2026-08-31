#include "guard_common.hpp"
// TileOP-API doc guard: TSORT (SFU irregular) — stable row-group sort.
// Source: docs/tileop-usage/sort.md — FULL signature given.
//   template <ValueDstTile, IndexDstTile, SourceTile>
//   void TSORT(valueDst, indexDst, source, sortWidth=32, descending=false);
// Contract: source/valueDst same FP16/FP32 dtype; indexDst U32; all Local VEC
//   RowMajor; dsts share Rows/Cols with source. Doc example uses 32x32.
int main() {
    constexpr int M = 32, N = 32, NE = M * N;
    float src[NE], val[NE];
    uint32_t idx[NE];
    gfill_seq(src, NE, 1.0f);
    gzero(val, NE);
    for (int i = 0; i < NE; ++i) idx[i] = 0;

    iter_t<float, M, N> gS((float *)src), gV(val);
    iter_t<uint32_t, M, N> gI(idx);
    auto gS0 = gS(0, 0);
    auto gV0 = gV(0, 0);
    auto gI0 = gI(0, 0);
    vtile_t<float, M, N> tS, tV;
    vtile_t<uint32_t, M, N> tI;
    TLOAD(tS, gS0);
    BENCHSTART;
    TSORT(tV, tI, tS);                 // width 32, ascending
    BENCHEND;
    TSTORE(gV0, tV);
    TSTORE(gI0, tI);
    return 0;
}
