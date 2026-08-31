#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL (CUBE) — D = A*B.
// Source: docs/tileop-usage/cube.md + matrix-postprocess.md.
//   基础类型: CubeTileM32<T,M,K> a; CubeTileN8<T,K,N> b;
//             CubeAccumulatorM32<AccT,M,N> out;  TMATMUL(out, a, b);
//   GM 边界用 TLOAD_CUBE / TSTORE_CUBE。
// NOTE(doc-gap): cube.md 说"用 TLOAD_CUBE/TSTORE_CUBE 做 GM 转换边界",但
//   docs/tileop-usage 未给这两个 wrapper 的 C++ 签名(参数个数/顺序/选择子)。
//   下面按最直觉的 (cube_tile, global_view) 形式尝试。
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hc[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    gzero(hc, M * N);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b);
    BENCHEND;
    TSTORE_CUBE(gC, out);
    return 0;
}
