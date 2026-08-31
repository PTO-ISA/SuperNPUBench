#include "guard_common.hpp"
// TileOP-API doc guard: chained RowMax + GroupMax + MaxAbs postprocess.
// Source: matrix-postprocess.md — "RowMax + GroupMax + MaxAbs"
//   fixp::keep_acc().row_max(in,out).group_max<8>(gout).max_abs().
//   .max_abs() 只能在已启用 RowMax 或 GroupMax 后调用;dst 紧凑顺序 D,RowMaxOut,GroupMaxOut.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hc[M * N], hrmi[M * 8];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    gzero(hc, M * N);
    for (int i = 0; i < M * 8; ++i) hrmi[i] = 0.0f;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> out;
    using RMTile = Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 1>;
    RMTile row_max_in, row_max_out;
    Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 4> group_max_out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);
    global_iterator<gm_t<float, 32, 8>, RMTile> gRMI(hrmi);
    auto gRMI0 = gRMI(0, 0);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(row_max_in, gRMI0);
    BENCHSTART;
    TMATMUL(out, a, b,
            fixp::keep_acc()
                .row_max(row_max_in, row_max_out)
                .group_max<8>(group_max_out)
                .max_abs());
    BENCHEND;
    TSTORE_CUBE(gC, out);
    return 0;
}
