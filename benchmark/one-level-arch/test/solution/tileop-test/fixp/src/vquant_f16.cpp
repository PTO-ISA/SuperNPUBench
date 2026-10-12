#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::vector<Mode>(tile) — vector quant parameter to FP16.
// Source: options.md（+ pto-spec matrix-postprocess.asl） — "Vector quant parameter"
//   fixp::vector<VQF322F16Pre>(quant) -> dst FP16; each 64-bit element same bit
//   layout as the scalar descriptor.
// Precision: res_check. F16 (floating) path, offset unused; golden pins pto-spec
//   matrix-postprocess.asl: dst = fp16(D * scale). FP19 16.0 per column.
static constexpr uint64_t make_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N], hd[M * N];
    uint64_t hq[2 * 32];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    for (int i = 0; i < M * N; ++i) hd[i] = (__half)0.0f;
    for (int i = 0; i < 2 * 32; ++i) hq[i] = make_quant(0x20C00u, 0);   // FP19 16.0

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<__half, M, N> out;
    // PTO-CUBE-AUX-CELLREG-001: logical [1,N] U64 on CUBE_N8 K2xN8.
    using QuantTile = CubeTileN8<uint64_t, 2, 32, 1, 32>;
    QuantTile quant;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<__half, RowMajor<M, N>> gD(hd);
    global_tensor<uint64_t, RowMajor<1, N>> gQ(hq);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD_CUBE(quant, gQ);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::vector<FixpPreQuantMode::VQF322F16Pre>(quant));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
