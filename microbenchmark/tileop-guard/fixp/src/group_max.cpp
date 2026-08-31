#include "guard_common.hpp"
// TileOP-API doc guard: fixp::keep_acc().group_max<GroupN>(out) — GroupMax reduction.
// Source: matrix-postprocess.md — "GroupMax"
//   group_max_tile: Tile<Vec, __fp32, 32, 8, RowMajor, 32, 4> — valid M x ceil(N/GroupN).
//   N=32, GroupN=8 -> valid columns = 4. dtype 精确匹配派生 AccType(FP32).
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
    // GroupMax out: 物理 32x8, valid 32x4 (N/GroupN = 32/8)
    Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 4> group_max_out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::keep_acc().group_max<8>(group_max_out));
    BENCHEND;
    TSTORE_CUBE(gC, out);
    return 0;
}
