#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::keep_acc().cscale(scale) — FP32 accumulator C scaling.
// Source: matrix-postprocess.md — "FP32 accumulator C scaling (PTO ISA 0.58.4)".
// Precision: res_check. pto-spec cube.asl MatrixInitialAccumulatorValue applies the
//   per-row U8 exponent to the initial accumulator C: TileProfileMatrixCScale =
//   C / 2^exponent (matrix-postprocess.asl). Then A@B accumulates on top:
//     d = A@B + C / 2^exp.  Here exp=1 (all rows) -> d = A@B + C/2. Golden = that.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half  ha[M * K], hb[K * N];
    float   hcc[M * N], hd[M * N];
    uint8_t hs[32 * 32];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    guard_read_bin(CHK_DIR "/in_c.bin", hcc, sizeof(hcc));
    gzero(hd, M * N);
    for (int i = 0; i < 32 * 32; ++i) hs[i] = 1;      // per-row exponent 1 -> C/2

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
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
