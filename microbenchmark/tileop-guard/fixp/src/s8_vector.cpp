#include "guard_common.hpp"
// TileOP-API doc guard: fixp::s8(tile) — vector-quant shortcut (VQF322S8Pre).
// Source: matrix-postprocess.md — "fixp::s8(quant) 是 VQF322S8Pre 的快捷形式".
static constexpr uint64_t make_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    int8_t hd[M * N];
    uint64_t hq[2 * 32];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = 0;
    for (int i = 0; i < 2 * 32; ++i) hq[i] = make_quant(0x40000u, 0);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<int8_t, M, N> out;
    using QuantTile = Tile<Location::Vec, uint64_t, 2, 32, BLayout::RowMajor, 1, 32>;
    QuantTile quant;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<int8_t, RowMajor<M, N>> gD(hd);
    global_iterator<gm_t<uint64_t, 2, 32>, QuantTile> gQ(hq);
    auto gQ0 = gQ(0, 0);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(quant, gQ0);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::s8(quant));   // vector-quant shortcut
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
