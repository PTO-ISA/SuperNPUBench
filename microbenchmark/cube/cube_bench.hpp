#ifndef CUBE_BENCH_HPP
#define CUBE_BENCH_HPP

// PTO ISA v0.58 single-PE CUBE microbench templates. Matrix operands and
// accumulators use persistent CELL layouts; GM transport uses TLOAD_CUBE and
// TSTORE_CUBE. There is intentionally no synthetic TLOAD_ND2NZ operation.

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>
#include "benchmark.h"
#include "bench_utils.hpp"

using namespace pto;

template <typename D>
using cube_acc_element_t = std::conditional_t<
    std::is_same_v<D, int8_t>, int32_t,
    std::conditional_t<std::is_same_v<D, uint8_t>, uint32_t, float>>;

template <typename D, int M, int K>
using cube_a_t = std::conditional_t<(M <= 16), CubeTileM16<D, M, K>,
                                    CubeTileM32<D, M, K>>;
template <typename D, int K, int N>
using cube_b_t = CubeTileN8<D, K, N>;
template <typename AccD, int M, int N>
using cube_c_t = std::conditional_t<(M <= 16), CubeAccumulatorM16<AccD, M, N>,
                                    CubeAccumulatorM32<AccD, M, N>>;
template <typename AccD, int N>
using cube_bias_t = Tile<Location::Bias, AccD, 8, N,
                         BLayout::RowMajor, 1, N>;

template <typename D, int M, int K>
using gm_a_t = global_tensor<D, RowMajor<M, K>>;
template <typename D, int K, int N>
using gm_b_t = global_tensor<D, RowMajor<K, N>>;
template <typename D, int M, int N>
using gm_c_t = global_tensor<D, RowMajor<M, N>>;

template <typename D, int M, int N, int K>
void bench_matmul(cube_acc_element_t<D> *out, D *a, D *b) {
    using AccD = cube_acc_element_t<D>;
    gm_a_t<D, M, K> gA(a); gm_b_t<D, K, N> gB(b); gm_c_t<AccD, M, N> gD(out);
    cube_a_t<D, M, K> tA; cube_b_t<D, K, N> tB; cube_c_t<AccD, M, N> tD;
    TLOAD_CUBE(tA, gA);
    TLOAD_CUBE(tB, gB);
    TMATMUL(tD, tA, tB);
    TSTORE_CUBE(gD, tD);
}

template <typename D, int M, int N, int K>
void bench_matmul_acc(cube_acc_element_t<D> *out,
                      cube_acc_element_t<D> *initial, D *a, D *b) {
    using AccD = cube_acc_element_t<D>;
    gm_a_t<D, M, K> gA(a); gm_b_t<D, K, N> gB(b);
    gm_c_t<AccD, M, N> gC(initial), gD(out);
    cube_a_t<D, M, K> tA; cube_b_t<D, K, N> tB;
    cube_c_t<AccD, M, N> tC, tD;
    TLOAD_CUBE(tA, gA);
    TLOAD_CUBE(tB, gB);
    TLOAD_CUBE(tC, gC);
    TMATMUL_ACC(tD, tC, tA, tB);
    TSTORE_CUBE(gD, tD);
}

template <typename D, int M, int N, int K>
void bench_matmul_bias(cube_acc_element_t<D> *out, D *a, D *b,
                       cube_acc_element_t<D> *bias) {
    using AccD = cube_acc_element_t<D>;
    gm_a_t<D, M, K> gA(a); gm_b_t<D, K, N> gB(b);
    gm_c_t<AccD, M, N> gD(out); global_tensor<AccD, RowMajor<1, N>> gBias(bias);
    cube_a_t<D, M, K> tA; cube_b_t<D, K, N> tB;
    cube_c_t<AccD, M, N> tD; cube_bias_t<AccD, N> tBias;
    TLOAD_CUBE(tA, gA);
    TLOAD_CUBE(tB, gB);
    TLOAD(tBias, gBias);
    TMATMUL_BIAS(tD, tA, tB, tBias);
    TSTORE_CUBE(gD, tD);
}

template <typename InD, typename AccD, int M, int N, int K>
void reference_matmul(AccD *out, InD *a, InD *b,
                      const AccD *initial = nullptr,
                      const AccD *bias = nullptr) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            AccD v = initial ? initial[m * N + n] : (AccD)0;
            if (bias) v += bias[n];
            for (int k = 0; k < K; ++k)
                v += (AccD)a[m * K + k] * (AccD)b[k * N + n];
            out[m * N + n] = v;
        }
    }
}

template <typename T>
constexpr T cube_epsilon() {
    if constexpr (std::is_integral_v<T>) return (T)0;
    return (T)1e-2;
}

#endif
