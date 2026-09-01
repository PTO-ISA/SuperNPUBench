#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::convert<Mode>() — generic parameter-free PreQuant.
// Source: matrix-postprocess.md — fixp::convert<FixpPreQuantMode::F322F16>() ->
//   PreQuant F322F16, dst FP16. Equivalent to fixp::f16(); exercises generic convert<>.
// Precision: res_check, golden = (A@B) cast to f16.
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN], hd[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<__half, GM, GN> out;    // dst FP16 per B.FPATR 表
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<__half, RowMajor<GM, GN>> gD(hd);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::convert<FixpPreQuantMode::F322F16>());
    BENCHEND;
    TSTORE_CUBE(gD, out);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
