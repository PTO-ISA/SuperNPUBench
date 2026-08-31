#include "guard_common.hpp"
// TileOP-API doc guard: TDEQUANT (SFU irregular) — S8/U8 -> FP32.
// Source: docs/tileop-usage/quant-and-im2col.md — FULL signature given.
//   template <RoundMode Mode=RNE, ...out, ...in>
//   void TDEQUANT(out &dst, in &src, float multiplier=1.0f, int32_t zeroPoint=0);
// Contract: dst FP32; src S8 or U8.
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    int8_t a[NE];
    float c[NE];
    for (int i = 0; i < NE; ++i) a[i] = (int8_t)((i % 127) - 64);
    gzero(c, NE);
    BENCHSTART;
    g_unary_cvt<int8_t, float, M, N>(c, a, [](auto& d, auto& s){
        TDEQUANT<RoundMode::RTZ>(d, s, 2.0f, 0);
    });
    BENCHEND;
    return 0;
}
