#include "guard_common.hpp"
// TileOP-API doc guard: fixp::convert<Mode>() — generic parameter-free PreQuant.
// Source: matrix-postprocess.md — "通用模式选择 / 无额外参数的转换"
//   fixp::convert<FixpPreQuantMode::F322F16>()  -> PreQuant F322F16, dst FP16.
// Equivalent to fixp::f16(); exercises the generic convert<> spelling.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N], hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = (__half)0.0f;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<__half, M, N> out;    // dst FP16 per B.FPATR 表

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<__half, RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::convert<FixpPreQuantMode::F322F16>());
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
