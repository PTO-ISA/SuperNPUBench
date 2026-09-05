#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::s8(uint64_t) — scalar quant to S8 (QF322S8Pre).
// Source: matrix-postprocess.md — "Scalar quant descriptor"
//   descriptor 布局(64-bit): FP19 scale [31:13], S8 offset(S9) [45:37].
// Precision: res_check. Golden pins pto-spec matrix-postprocess.asl:
//   D=A@B (fp32); act=D*scale; S9 round+sat -> +offset -> encode S8 (RNE).
//   scale FP19 0x20C00 = 16.0 (D*16 in +-40, rich non-saturating S8), offset 5.
static constexpr uint64_t make_s8_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    int8_t hd[M * N];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    for (int i = 0; i < M * N; ++i) hd[i] = 0;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<int8_t, M, N> out;    // dst S8 per B.FPATR 表

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<int8_t, RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    const uint64_t quant_desc = make_s8_quant(0x20C00u, 5);   // FP19 16.0, offset 5
    BENCHSTART;
    TMATMUL(out, a, b, fixp::s8(quant_desc));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
