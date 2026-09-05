#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL + RowMax postprocess.
// Source: matrix-postprocess.md — fixp::keep_acc().row_max(row_max_out).
//   row_max_tile: Tile<Vec, float, 32, 8, RowMajor, 32, 1> (valid M x 1, 物理 >=128B).
//   RowMaxEn=1, RowMaxInit=0. dtype 必须精确匹配派生 AccType(FP32)。
// Precision: res_check, golden checks the accumulator out = A@B (row_max side
// output is not dumped).
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN];
static float  hc[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<float, GM, GN> out;
    Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 1> row_max_out;
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<float,  RowMajor<GM, GN>> gC(hc);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::keep_acc().row_max(row_max_out));
    BENCHEND;
    TSTORE_CUBE(gC, out);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
