#include "guard_common.hpp"
// TileOP-API doc guard: fixp::s8(desc).lrelu(fp19) — scalar quant + scalar LReLU.
// Source: matrix-postprocess.md — "LReLU"
//   scalar quant via B.IOR SrcReg0, LReLU via B.IOR SrcReg1; lrelu_fp19 low 19 bits.
static constexpr uint64_t make_s8_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    int8_t hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = 0;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<int8_t, M, N> out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<int8_t, RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    const uint64_t quant_desc = make_s8_quant(0x40000u, 0);
    const uint64_t lrelu_fp19 = 0x10000u & 0x7ffffu;   // FP19 slope in low 19 bits
    BENCHSTART;
    TMATMUL(out, a, b, fixp::s8(quant_desc).lrelu(lrelu_fp19));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
