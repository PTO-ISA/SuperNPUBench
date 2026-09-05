#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL (CUBE) — D = A*B.
// Source: docs/tileop-usage/cube.md + matrix-postprocess.md.
//   CubeTileM32<T,M,K> a; CubeTileN8<T,K,N> b; CubeAccumulatorM32<AccT,M,N> out;
//   TMATMUL(out,a,b); GM 边界用 TLOAD_CUBE/TSTORE_CUBE。
// NOTE(doc-gap): TLOAD_CUBE/TSTORE_CUBE 无 C++ 签名;(cube_tile, global_view) 推断。
// Precision: res_check, f16 A/B host-generated, independent numpy golden (f32 @).
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN];
static float  hc[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<float, GM, GN> out;
    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<float,  RowMajor<GM, GN>> gC(hc);
    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b);
    BENCHEND;
    TSTORE_CUBE(gC, out);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
