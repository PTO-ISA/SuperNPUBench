#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::f16().prelu(tile) — convert + PReLU (no quant).
// Source: matrix-postprocess.md — "PReLU" (length-N FP19 Tile, low 19 bits = slope).
// Precision: res_check. fixp::f16() convert has scale 1.0; pto-spec
//   matrix-postprocess.asl multiplier: value>=0 -> 1.0, value<0 -> slope. F16
//   floating encode: dst = fp16(where(D>=0, D, D*slope)). Slope FP19 0.5 per column.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N], hd[M * N];
    uint64_t hp[2 * 32];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    for (int i = 0; i < M * N; ++i) hd[i] = (__half)0.0f;
    for (int i = 0; i < 2 * 32; ++i) hp[i] = 0x1F800u & 0x7ffffu;   // FP19 slope 0.5

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
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
