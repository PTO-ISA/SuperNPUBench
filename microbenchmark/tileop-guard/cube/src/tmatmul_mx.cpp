#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL_MX (CUBE) — FP16/BF16 pair, no scales.
// Source: matrix-postprocess.md — TMATMUL_MX(Dst, A, B, options); // no scales
//   MX 的 FP16/BF16 侧不得提供 scale；便捷重载省去 scale operand。
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    gzero(hd, M * N);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> d;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL_MX(d, a, b, fixp::keep_acc());   // FP16 pair: no scales
    BENCHEND;
    TSTORE_CUBE(gD, d);
    return 0;
}
