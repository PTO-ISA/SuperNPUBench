#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL_ACC (CUBE) — D = C + A*B.
// Source: matrix-postprocess.md — TMATMUL_ACC(Dst, C, A, B, options);
//   .ACC 读显式累加器 C 作首源，写独立 D。无 options 等价 keep_acc()。
// Precision: res_check, host-generated A/B(f16)+C(f32), golden D = C + A@B.
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN];
static float  hcc[GM * GN], hd[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    guard_read_bin(CHK_DIR "/in_c.bin", hcc, sizeof(hcc));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<float, GM, GN> c, d;
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<float,  RowMajor<GM, GN>> gC(hcc);
    global_tensor<float,  RowMajor<GM, GN>> gD(hd);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD_CUBE(c, gC);
    BENCHSTART;
    TMATMUL_ACC(d, c, a, b, fixp::keep_acc());
    BENCHEND;
    TSTORE_CUBE(gD, d);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
