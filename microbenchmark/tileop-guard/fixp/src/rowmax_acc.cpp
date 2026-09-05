#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: fixp::keep_acc().row_max(in, out) — accumulate existing RowMax.
// Source: matrix-postprocess.md — "累加已有 RowMax" (RowMaxEn=1, RowMaxInit=1).
// Precision: res_check. keep_acc + RowMax publishes the *auxiliary* RowMaxOut to a
//   separate destination; the main D committed to gC is the plain fp32 matmul
//   (pto-spec postprocess.asl: pre_quant_mode==0 -> identity on D). Golden = A@B.
//   RowMaxIn is host-zero (unused by the main-D golden).
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hc[M * N], hrmi[M * 8];
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
    guard_read_bin(CHK_DIR "/in_b.bin", hb, sizeof(hb));
    gzero(hc, M * N);
    for (int i = 0; i < M * 8; ++i) hrmi[i] = 0.0f;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<float, M, N> out;
    using RMTile = Tile<Location::Vec, float, 32, 8, BLayout::RowMajor, 32, 1>;
    RMTile row_max_in, row_max_out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<float,  RowMajor<M, N>> gC(hc);
    global_iterator<gm_t<float, 32, 8>, RMTile> gRMI(hrmi);
    auto gRMI0 = gRMI(0, 0);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(row_max_in, gRMI0);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::keep_acc().row_max(row_max_in, row_max_out));
    BENCHEND;
    TSTORE_CUBE(gC, out);
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
    return 0;
}
