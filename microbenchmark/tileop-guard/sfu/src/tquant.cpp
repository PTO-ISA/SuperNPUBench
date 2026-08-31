#include "guard_common.hpp"
// TileOP-API doc guard: TQUANT (SFU irregular) — FP32 -> S8/U8.
// Source: docs/tileop-usage/quant-and-im2col.md — FULL signature given.
//   template <RoundMode Mode=RNE, bool Saturate=false, ...out, ...in>
//   void TQUANT(out &dst, in &src, float multiplier=1.0f, int32_t zeroPoint=0);
// Contract: src FP32; dst S8 or U8 (DataType derived from dst).
int main() {
    constexpr int M = 8, N = 256, NE = M * N;
    float a[NE];
    int8_t c[NE];
    gfill_seq(a, NE); for (int i = 0; i < NE; ++i) c[i] = 0;
    BENCHSTART;
    g_unary_cvt<float, int8_t, M, N>(c, a, [](auto& d, auto& s){
        TQUANT<RoundMode::RNE, /*saturate=*/true>(d, s, 0.5f, 1);
    });
    BENCHEND;
    return 0;
}
