#include "guard_common.hpp"
// TileOP-API doc guard: TMATMUL_BIAS (CUBE) — D = A*B + Bias.
// Source: matrix-postprocess.md — TMATMUL_BIAS<Attr>(Dst, A, B, Bias, options).
//   "辅助参数仍是普通 Local Tile"(line 13);static_assert 反馈 Bias 必须
//   匹配 derived AccType(fp16 输入→FP32)且为 ordinary RowMajor layout。
// NOTE(doc-gap): 文档未明确 Bias 的 dtype=派生AccType + RowMajor + **valid
//   shape 必须 1 x N**(非 M x N),且用普通 TLOAD(非 TLOAD_CUBE);全靠编译器
//   static_assert 逐条反推。matrix-postprocess.md line 13 只说"辅助参数是普通
//   Local Tile",这三条约束一条都没写。
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    float  hbias[N], hc[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < N; ++i) hbias[i] = 0.5f;
    gzero(hc, M * N);

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    vtile_t<float, 1, N> bias;                    // ordinary RowMajor FP32, 1 x N
    CubeAccumulatorM32<float, M, N> out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    iter_t<float, 1, N> gBias(hbias);
    auto gBias0 = gBias(0, 0);
    global_tensor<float,  RowMajor<M, N>> gC(hc);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    TLOAD(bias, gBias0);                           // ordinary Local tile load
    BENCHSTART;
    TMATMUL_BIAS(out, a, b, bias, fixp::keep_acc());
    BENCHEND;
    TSTORE_CUBE(gC, out);
    return 0;
}
