#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL_BIAS (CUBE) — D = A*B + Bias.
// Source: matrix-postprocess.md — TMATMUL_BIAS<Attr>(Dst, A, B, Bias, options).
// NOTE(doc-gap): Bias 必须 派生AccType(FP32) + ordinary RowMajor + valid 1 x N,
//   且用普通 TLOAD(非 TLOAD_CUBE);全靠 static_assert 反推(文档只说"普通 Local Tile")。
// Precision: res_check, golden = A@B + bias(1xN broadcast).
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN];
static float  hbias[GN], hc[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    guard_read_bin(CHK_DIR "/in_bias.bin", hbias, sizeof(hbias));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    vtile_t<float, 1, GN> bias;                    // ordinary RowMajor FP32, 1 x N
    CubeAccumulatorM32<float, GM, GN> out;
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    iter_t<float, 1, GN> gBias(hbias);
    auto gBias0 = gBias(0, 0);
    global_tensor<float,  RowMajor<GM, GN>> gC(hc);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(bias, gBias0);                            // ordinary Local tile load
    BENCHSTART;
    TMATMUL_BIAS(out, a, b, bias, fixp::keep_acc());
    BENCHEND;
    TSTORE_CUBE(gC, out);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
