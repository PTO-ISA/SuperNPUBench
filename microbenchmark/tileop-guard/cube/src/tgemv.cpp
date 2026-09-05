#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API doc guard: TGEMV (CUBE) — D(1,N) = Vec(1,K) * Mtx(K,N), M=1.
// Source: matrix-postprocess.md — TGEMV(Dst, Mtx, Vec, options);
//   Vec=CubeTileM16<T,1,K>, Mtx=CubeTileN8<T,K,N>, Dst=CubeAccumulatorM16<AccT,1,N>.
//   数学源顺序 A=Vec, B=Mtx；Local-only（任何 B.IOS illegal）。
// Precision: res_check, f16 vec/mtx host-generated, independent numpy golden
//   (fam='matmul' M=1: D = vec(1,K) @ mtx(K,N)); in_a=vec, in_b=mtx.
constexpr int GN = 32, GK = 32;
static __half hv[1 * GK], hm[GK * GN];
static float  hd[1 * GN];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", hv, sizeof(hv));
    guard_read_bin(CHK_DIR "/in_b.bin", hm, sizeof(hm));
    CubeTileM16<__half, 1, GK> vec;
    CubeTileN8<__half, GK, GN>  mtx;
    // doc says Dst = CubeAccumulatorM16<AccT,1,N>. gfrun 断言要求物理 col>=N
    // 且维度为 2 的幂、dst 容量 >= M x N 输出区。M=1,N=32 已是 2 的幂。
    CubeAccumulatorM16<float, 1, GN> d;

    global_tensor<__half, RowMajor<1, GK>> gV(hv);
    global_tensor<__half, RowMajor<GK, GN>> gM(hm);
    global_tensor<float,  RowMajor<1, GN>> gD(hd);

    TLOAD_CUBE(vec, gV);
    TLOAD_CUBE(mtx, gM);
    BENCHSTART;
    TGEMV(d, mtx, vec, fixp::keep_acc());   // doc arg order: (Dst, Mtx, Vec, options)
    BENCHEND;
    TSTORE_CUBE(gD, d);
    guard_dump_bin(CHK_DIR "/out.bin", hd, sizeof(hd));
    return 0;
}
