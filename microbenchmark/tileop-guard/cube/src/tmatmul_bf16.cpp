#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL + fixp::bf16() — D = A*B, PreQuant F322BF16.
// Source: matrix-postprocess.md line 164 — TMATMUL(dst_bf16, a, b, fixp::bf16());
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    __bf16 hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = (__bf16)0.0f;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<__bf16, M, N> out;    // dst dtype BF16 per options 表

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<__bf16, RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::bf16());
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
