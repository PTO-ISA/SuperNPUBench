#include "guard_common.hpp"
// TileOP-API doc guard: fixp::keep_acc().cscale(scale) — FP32 accumulator C scaling.
// Source: matrix-postprocess.md — "FP32 accumulator C scaling (PTO ISA 0.58.4)"
//   只适用于 TMATMUL_ACC 的 FP32 accumulator 路径.
//   CScale: Tile<Vec, uint8_t, 32, 32, CubeM32, 32, 1> — Local U8 CUBE_M32, valid M x 1.
// NOTE(doc-gap): 文档给出 cscale(scale) 调用和 CScale tile 类型,但未说明该 CUBE_M32
//   layout tile 的加载方式;这里按 CUBE tile 惯例用 TLOAD_CUBE 尝试.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half  ha[M * K], hb[K * N];
    float   hcc[M * N], hd[M * N];
    uint8_t hs[32 * 32];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hcc[i] = 1.0f;
    gzero(hd, M * N);
    for (int i = 0; i < 32 * 32; ++i) hs[i] = 1;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> c, d;
    // CScale: Local U8 CUBE_M32, valid M x 1
    Tile<Location::Vec, uint8_t, 32, 32, BLayout::CubeM32, 32, 1> scale;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hcc);
    global_tensor<float,  RowMajor<M, N>> gD(hd);
    global_tensor<uint8_t, RowMajor<32, 32>> gS(hs);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD_CUBE(c, gC);
    TLOAD_CUBE(scale, gS);
    BENCHSTART;
    TMATMUL_ACC(d, c, a, b, fixp::keep_acc().cscale(scale));
    BENCHEND;
    TSTORE_CUBE(gD, d);
    return 0;
}
