#include "guard_common.hpp"
// TileOP-API doc guard: fixp::f16().prelu(tile) — convert + PReLU.
// Source: matrix-postprocess.md — "PReLU"
//   PReLU 参数是长度 N 的 FP19 Tile: Tile<Vec, uint64_t, 2, 32, RowMajor, 1, 32>,
//   每 element 低 19 bit 保存 FP19 slope. 可配合无 quant 的 convert(fixp::f16()).
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N], hd[M * N];
    uint64_t hp[2 * 32];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = (__half)0.0f;
    for (int i = 0; i < 2 * 32; ++i) hp[i] = 0x10000u & 0x7ffffu;   // FP19 slope

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<__half, M, N> out;
    using Fp19Tile = Tile<Location::Vec, uint64_t, 2, 32, BLayout::RowMajor, 1, 32>;
    Fp19Tile prelu;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<__half, RowMajor<M, N>> gD(hd);
    global_iterator<gm_t<uint64_t, 2, 32>, Fp19Tile> gP(hp);
    auto gP0 = gP(0, 0);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(prelu, gP0);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::f16().prelu(prelu));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
