#include "guard_common.hpp"
// TileOP-API doc guard: TGEMV (CUBE) — D = Vec(1,K) * Mtx(K,N), M=1.
// Source: matrix-postprocess.md — TGEMV(Dst, Mtx, Vec, options);
//   Vec=CubeTileM16<T,1,K>, Mtx=CubeTileN8<T,K,N>, Dst=CubeAccumulatorM16<AccT,1,N>.
//   数学源顺序 A=Vec, B=Mtx；Local-only（任何 B.IOS illegal）。
int main() {
    constexpr int N = 32, K = 32;
    __half hv[1 * K], hm[K * N];
    float  hd[1 * N];
    for (int i = 0; i < K; ++i) hv[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hm[i] = (__half)(0.02f * i);
    gzero(hd, N);

    CubeTileM16<__half, 1, K> vec;
    CubeTileN8<__half, K, N>  mtx;
    // doc says Dst = CubeAccumulatorM16<AccT,1,N>. gfrun 断言要求物理 col>=N
    // 且维度为 2 的幂、dst 容量 >= M x N 输出区。M=1,N=32 已是 2 的幂。
    CubeAccumulatorM16<float, 1, N> d;

    global_tensor<__half, RowMajor<1, K>> gV(hv);
    global_tensor<__half, RowMajor<K, N>> gM(hm);
    global_tensor<float,  RowMajor<1, N>> gD(hd);

    TLOAD_CUBE(vec, gV);
    TLOAD_CUBE(mtx, gM);
    BENCHSTART;
    TGEMV(d, mtx, vec, fixp::keep_acc());   // doc arg order: (Dst, Mtx, Vec, options)
    BENCHEND;
    TSTORE_CUBE(gD, d);
    return 0;
}
