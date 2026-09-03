#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TMATMUL_MX (CUBE) — FP16/BF16 pair, no scales.
// Source: matrix-postprocess.md — TMATMUL_MX(Dst, A, B, options); // no scales
//   MX 的 FP16/BF16 侧不得提供 scale；便捷重载省去 scale operand。
// Precision: res_check, f16 A/B host-generated, independent numpy golden
//   (fam='matmul': D = A(M,K) @ B(K,N)); MX f16 pair == plain matmul math.
constexpr int GM = 32, GN = 32, GK = 32;
static __half ha[GM * GK], hb[GK * GN];
static float  hd[GM * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    CubeTileM32<__half, GM, GK> a;
    CubeTileN8<__half, GK, GN>  b;
    CubeAccumulatorM32<float, GM, GN> d;

    global_tensor<__half, RowMajor<GM, GK>> gA(ha);
    global_tensor<__half, RowMajor<GK, GN>> gB(hb);
    global_tensor<float,  RowMajor<GM, GN>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL_MX(d, a, b, fixp::keep_acc());   // FP16 pair: no scales
    BENCHEND;
    TSTORE_CUBE(gD, d);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
