#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL + RowMax postprocess.
// Source: matrix-postprocess.md — fixp::keep_acc().row_max(row_max_out).
//   row_max_tile: Tile<Vec, __fp32, 32, 8, RowMajor, 32, 1> (valid M x 1, 物理 >=128B).
//   设置 RowMaxEn=1, RowMaxInit=0；无 RowMaxIn。dtype 必须精确匹配派生 AccType(FP32)。
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hc[M * N], hrm[M * 8];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    gzero(hc, M * N); gzero(hrm, M * 8);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> out;
    // row_max 输出: 物理 32x8 (>=128B), valid 32x1
    Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 1> row_max_out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::keep_acc().row_max(row_max_out));
    BENCHEND;
    TSTORE_CUBE(gC, out);
    return 0;
}
