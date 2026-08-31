#include "guard_common.hpp"
// TileOP-API doc guard: fixp::scalar<Mode>(descriptor) — generic scalar-param spelling.
// Source: matrix-postprocess.md — "通用模式选择 / 需要 scalar descriptor 的模式"
//   fixp::scalar<Mode>(descriptor); 与快捷式 fixp::s8(desc)==scalar<QF322S8Pre> 对应.
//   本 demo 用 QF322S8Pre 演示 generic 写法本身(dst S8).
// NOTE(doc-gap): B.FPATR 表(matrix-postprocess.md)列 QF322S16Pre->S16,但用 fp16
//   矩阵输入 + S16 dst 会触发 static_assert "PreQuantMode incompatible with the
//   derived matrix accumulator type" / "D dtype must match derived accumulator".
//   表未给 PreQuantMode 与输入矩阵 dtype 的兼容矩阵(S16 量化疑似要求整数矩阵输入).
static constexpr uint64_t make_s8_quant(uint32_t fp19_scale, int16_t offset) {
    return (static_cast<uint64_t>(fp19_scale & 0x7ffff) << 13) |
           ((static_cast<uint64_t>(offset) & 0x1ff) << 37);
}
int main() {
    constexpr int M = 32, N = 32, K = 32;
    __half ha[M * K], hb[K * N];
    int8_t hd[M * N];
    for (int i = 0; i < M * K; ++i) ha[i] = (__half)(0.01f * i);
    for (int i = 0; i < K * N; ++i) hb[i] = (__half)(0.02f * i);
    for (int i = 0; i < M * N; ++i) hd[i] = 0;

    CubeTileM32<__half, M, K> a;
    CubeTileN8<__half, K, N>  b;
    CubeAccumulatorM32<int8_t, M, N> out;

    global_tensor<__half, RowMajor<M, K>> gA(ha);
    global_tensor<__half, RowMajor<K, N>> gB(hb);
    global_tensor<int8_t, RowMajor<M, N>> gD(hd);

    TLOAD_CUBE(a, gA);
    TLOAD_CUBE(b, gB);
    const uint64_t desc = make_s8_quant(0x40000u, 0);
    BENCHSTART;
    TMATMUL(out, a, b, fixp::scalar<FixpPreQuantMode::QF322S8Pre>(desc));
    BENCHEND;
    TSTORE_CUBE(gD, out);
    return 0;
}
