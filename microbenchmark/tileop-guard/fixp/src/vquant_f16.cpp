#include "guard_common.hpp"
// TileOP-API doc guard: fixp::vector<Mode>(tile) — vector quant parameter.
// Source: matrix-postprocess.md — "Vector quant parameter"
//   quant Tile: Tile<Vec, uint64_t, 2, 32, RowMajor, 1, 32> (物理>=128B, valid 1xN).
//   fixp::vector<FixpPreQuantMode::VQF322F16Pre>(quant) -> dst FP16.
//   每个 64-bit element 与 scalar descriptor 同 bit layout.
static constexpr uint64_t make_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N], hd[M * N];
    uint64_t hq[2 * 32];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = (__half)0.0f;
    for (int i = 0; i < 2 * 32; ++i) hq[i] = make_quant(0x40000u, 0);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<__half, M, N> out;
    // vector quant parameter tile: 物理 2x32 (>=128B), valid 1x32
    using QuantTile = Tile<Location::Vec, uint64_t, 2, 32, BLayout::RowMajor, 1, 32>;
    QuantTile quant;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<__half, RowMajor<M, N>> gD(hd);
    global_iterator<gm_t<uint64_t, 2, 32>, QuantTile> gQ(hq);
    auto gQ0 = gQ(0, 0);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(quant, gQ0);                 // ordinary Local tile load
    BENCHSTART;
    TMATMUL(out, a, b, fixp::vector<FixpPreQuantMode::VQF322F16Pre>(quant));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
