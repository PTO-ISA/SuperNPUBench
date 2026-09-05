#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::keep_acc().group_max<GroupN>(out) — GroupMax reduction.
// Source: matrix-postprocess.md — "GroupMax" (N=32, GroupN=8 -> valid columns = 4).
// Precision: res_check. GroupMaxOut goes to a separate auxiliary destination; the
//   main D committed to gC is the plain fp32 matmul (postprocess.asl identity on D
//   for pre_quant_mode==0). Golden = A@B.
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hc[M * N];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    gzero(hc, M * N);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> out;
    // GroupMax out: 物理 32x8, valid 32x4 (N/GroupN = 32/8)
    Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 4> group_max_out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::keep_acc().group_max<8>(group_max_out));
    BENCHEND;
    TSTORE_CUBE(gC, out);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
