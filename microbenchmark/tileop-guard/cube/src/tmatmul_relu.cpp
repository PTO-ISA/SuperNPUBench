#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL + fixp::f16().relu() — PreQuant + ReLU 链式。
// Source: matrix-postprocess.md line 178 — TMATMUL(dst_fp16, a, b, fixp::f16().relu());
// Precision: res_check, golden = relu(A@B) cast to f16.
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN], hd[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<__half, GM, GN> out;
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<__half, RowMajor<GM, GN>> gD(hd);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::f16().relu());
    BENCHEND;
    TSTORE_CUBE(gD, out);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
