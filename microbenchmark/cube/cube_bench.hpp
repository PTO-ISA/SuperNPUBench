#ifndef CUBE_BENCH_HPP
#define CUBE_BENCH_HPP

// CUBE matrix micro-bench templates.
// Intrinsic naming follows DavinciOO/PTO (OPERATOR_REFERENCE.md §8):
//   TMATMUL / TMATMUL_ACC / TMATMUL_BIAS / TMATMUL_MX
//   TGEMV  / TGEMV_ACC  / TGEMV_BIAS  / TGEMV_MX
//   ACCCVT  (ACC -> ordinary Tile export)
// ACC destination follows the CUBE/ACC contract; export via ACCCVT.

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "bench_utils.hpp"

using namespace pto;

template <typename D, int M, int K>
using gmA_t = global_tensor<D, RowMajor<M, K>>;
template <typename D, int K, int N>
using gmB_t = global_tensor<D, RowMajor<K, N>>;
template <typename D, int M, int N>
using gmC_t = global_tensor<D, RowMajor<M, N>>;

template <typename D, int M, int K>
using tL_t = TileLeft<D, M, K>;
template <typename D, int K, int N>
using tR_t = TileRight<D, K, N>;
template <typename D, int M, int N>
using tAcc_t = TileAcc<float, M, N>;  // ACC is float accumulator regardless of input dtype
template <typename D, int M, int N>
using tOut_t = Tile<Location::Vec, D, M, N, BLayout::RowMajor>;

// C = A * B   (TMATMUL -> ACC -> ACCCVT -> Tile -> GM)
template <typename D, int M, int N, int K>
void bench_matmul(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TMATMUL(tAcc, tA, tB);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C += A * B  (TMATMUL_ACC accumulate into ACC)
template <typename D, int M, int N, int K>
void bench_matmul_acc(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TMATMUL_ACC(tAcc, tA, tB);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C = A * B + bias  (TMATMUL_BIAS)
template <typename D, int M, int N, int K>
void bench_matmul_bias(D *c, D *a, D *b, D *bias) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    using itBias = global_iterator<gmC_t<D, 1, N>, tOut_t<D, 1, N>>;
    itA gA(a); itB gB(b); itC gC(c); itBias gBias(bias);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0), gBias0 = gBias(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB; tOut_t<D, 1, N> tBias;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tBias, gBias0);
    TMATMUL_BIAS(tAcc, tA, tB, tBias);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C = A * B  microscaling (TMATMUL_MX, with per-tile scale factors)
template <typename D, int M, int N, int K>
void bench_matmul_mx(D *c, D *a, D *as, D *b, D *bs) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a), gAs(as); itB gB(b), gBs(bs); itC gC(c);
    auto gA0 = gA(0, 0), gAs0 = gAs(0, 0), gB0 = gB(0, 0), gBs0 = gBs(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA, tAs; tR_t<D, K, N> tB, tBs;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0); TLOAD(tAs, gAs0);
    TLOAD(tB, gB0); TLOAD(tBs, gBs0);
    TMATMUL_MX(tAcc, tA, tAs, tB, tBs);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// FIXME: TGEMV / TGEMV_ACC / TGEMV_BIAS / TGEMV_MX are not yet exposed by the
// toolchain (only TMATMUL* and ACCCVT are). The gemv bench templates are
// disabled until the TGEMV* intrinsics land. Re-enable and regenerate gemv
// cases when they appear.
#if 0
// C = A * v  (TGEMV matrix-vector)
template <typename D, int M, int N, int K>
void bench_gemv(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TGEMV(tAcc, tA, tB);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C += A * v  (TGEMV_ACC)
template <typename D, int M, int N, int K>
void bench_gemv_acc(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TGEMV_ACC(tAcc, tA, tB);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C = A * v + bias  (TGEMV_BIAS)
template <typename D, int M, int N, int K>
void bench_gemv_bias(D *c, D *a, D *b, D *bias) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    using itBias = global_iterator<gmC_t<D, 1, N>, tOut_t<D, 1, N>>;
    itA gA(a); itB gB(b); itC gC(c); itBias gBias(bias);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0), gBias0 = gBias(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB; tOut_t<D, 1, N> tBias;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tBias, gBias0);
    TGEMV_BIAS(tAcc, tA, tB, tBias);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

// C = A * v  microscaling (TGEMV_MX, with per-tile scale factors)
template <typename D, int M, int N, int K>
void bench_gemv_mx(D *c, D *a, D *as, D *b, D *bs) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a), gAs(as); itB gB(b), gBs(bs); itC gC(c);
    auto gA0 = gA(0, 0), gAs0 = gAs(0, 0), gB0 = gB(0, 0), gBs0 = gBs(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA, tAs; tR_t<D, K, N> tB, tBs;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0); TLOAD(tAs, gAs0);
    TLOAD(tB, gB0); TLOAD(tBs, gBs0);
    TGEMV_MX(tAcc, tA, tAs, tB, tBs);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}
#endif  // FIXME: TGEMV* not exposed yet

// standalone ACCCVT: load A, matmul to ACC, export via ACCCVT
template <typename D, int M, int N, int K>
void bench_acccvt(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TMATMUL(tAcc, tA, tB);
    ACCCVT(tOut, tAcc);
    TSTORE(gC0, tOut);
}

#endif
