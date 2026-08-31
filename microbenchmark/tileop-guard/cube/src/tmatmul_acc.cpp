#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL_ACC (CUBE) — D = C + A*B.
// Source: matrix-postprocess.md — TMATMUL_ACC(Dst, C, A, B, options);
//   .ACC 读显式累加器 C 作首源，写独立 D。无 options 等价 keep_acc()。
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hcc[M * N], hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hcc[i] = 1.0f;
    gzero(hd, M * N);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> c, d;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hcc);
    global_tensor<float,  RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD_CUBE(c, gC);
    BENCHSTART;
    TMATMUL_ACC(d, c, a, b, fixp::keep_acc());
    BENCHEND;
    TSTORE_CUBE(gD, d);
    return 0;
}
