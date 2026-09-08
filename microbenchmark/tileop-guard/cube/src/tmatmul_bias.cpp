#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL_BIAS (CUBE) — D = A*B + Bias.
// Source: options.md（+ pto-spec matrix-postprocess.asl） — TMATMUL_BIAS<Attr>(Dst, A, B, Bias, options).
// NOTE(doc-gap 已修复, API 0566283): 早期文档只说 Bias 是"普通 Local Tile",约束靠 static_assert 反推。
//   现 TMATMUL_BIAS.md:51-56 已明确:普通 RowMajor Tile + dtype=FP32(同 accumulator) + valid 固定 1×N +
//   用普通 TLOAD(非 TLOAD_CUBE)。**bias 已对齐文档示例 Location::Bias(2026-09-08)**:spec TMATMUL_BIAS.asl
//   只要求"ordinary Local row-major 1xN accumulator型"(未限 location);Location::Vec 与 Location::Bias 经
//   指令级 diff 证发射 tile bundle 完全相同(bias role 由参数位置定),两者等价,此处采用文档示例的 Location::Bias。
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
    Tile<Location::Bias, float, 1, GN, BLayout::RowMajor> bias;  // 对齐 TMATMUL_BIAS.md 示例 Location::Bias
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
