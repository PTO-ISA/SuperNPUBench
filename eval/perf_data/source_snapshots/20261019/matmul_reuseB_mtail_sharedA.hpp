#pragma once

#include "matmul_reuseB.hpp"

// Test variant for M=2*128+32, full N tile: keep the M tail in the same
// K chain as the two full-M accumulators so all three consume one B load.
namespace matmul_reuse_b_mtail_shared_a_detail {

template <bool FirstK, bool LastK, int ValidK, typename dtype,
          int gM, int gN, int gK, int tN, int tK>
PTO_SHARED_INLINE void run_k(
    dtype *a_ptr, dtype *b_ptr,
    CubeAccumulatorM32<float, 32, tN> &c0,
    CubeAccumulatorM32<float, 32, tN> &c1,
    CubeAccumulatorM16<float, 16, tN, 16, tN> &c_tail,
    int n_begin, int k_begin, int tid) {
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA = CubeTileM32<dtype, 32, tK, 32, ValidK>;
    using tileATail = SharedMatrixLeft<dtype, 64, tK, 32, ValidK>;
    using tileB = SharedMatrixRight<dtype, tN, tK, tN, ValidK>;
    constexpr auto options = fixp::keep_acc();

    SharedTile<tileB> b;
    gmB gb(b_ptr + n_begin * gK + k_begin);
    TLOAD<tileB, 1>(b, gb);
    auto run_full = [&](auto &c, int m_begin) {
        tileA a;
        gmA ga(a_ptr + (m_begin + tid * 32) * gK + k_begin);
        TLOAD_CUBE(a, ga);
        if constexpr (FirstK) {
            if constexpr (LastK) {
                TMATMUL(c, a, b, options, 128);
            } else {
                TMATMUL(c, a, b, options.raw_acc(), 128);
            }
        } else if constexpr (LastK) {
            TMATMUL_ACC(c, c, a, b, options.acc_hint(), 128);
        } else {
            TMATMUL_ACC(c, c, a, b, options.raw_acc().acc_hint(), 128);
        }
    };
    run_full(c0, 0);
    run_full(c1, 128);

    SharedTile<tileATail> a_tail;
    gmA ga_tail(a_ptr + 256 * gK + k_begin);
    TLOAD<tileATail, 1>(a_tail, ga_tail);
    if constexpr (FirstK) {
        if constexpr (LastK) {
            TMATMUL(c_tail, a_tail, b, options);
        } else {
            TMATMUL(c_tail, a_tail, b, options.raw_acc());
        }
    } else if constexpr (LastK) {
        TMATMUL_ACC(c_tail, c_tail, a_tail, b, options.acc_hint());
    } else {
        TMATMUL_ACC(c_tail, c_tail, a_tail, b,
                    options.raw_acc().acc_hint());
    }
}

template <int RemainingK, int OffsetK, bool FirstK, typename dtype,
          int gM, int gN, int gK, int tN, int tK>
PTO_SHARED_INLINE void run_k_tail(
    dtype *a_ptr, dtype *b_ptr,
    CubeAccumulatorM32<float, 32, tN> &c0,
    CubeAccumulatorM32<float, 32, tN> &c1,
    CubeAccumulatorM16<float, 16, tN, 16, tN> &c_tail,
    int n_begin, int tid) {
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk <= tK);
    run_k<FirstK, RemainingK == chunk, chunk, dtype, gM, gN, gK, tN, tK>(
        a_ptr, b_ptr, c0, c1, c_tail, n_begin, OffsetK, tid);
    if constexpr (RemainingK > chunk) {
        run_k_tail<RemainingK - chunk, OffsetK + chunk, false,
                   dtype, gM, gN, gK, tN, tK>(
            a_ptr, b_ptr, c0, c1, c_tail, n_begin, tid);
    }
}

}  // namespace matmul_reuse_b_mtail_shared_a_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseB_mtail_sharedA(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    static_assert(gM == 288 && tM == 128, "test variant requires M288/tM128");
    static_assert(gN % tN == 0, "test variant requires full N tiles");
    static_assert(gK > tK && gK % tK == 96,
                  "test variant requires a 96-column K tail");
    static_assert(tK >= 128 && (tK & (tK - 1)) == 0);
    using tileA = CubeTileM32<dtype, 32, tK>;
    using tileATail = SharedMatrixLeft<dtype, 64, tK, 32, tK>;
    using tileB = SharedMatrixRight<dtype, tN, tK>;
    using cFull = CubeAccumulatorM32<float, 32, tN>;
    using cTail = CubeAccumulatorM16<float, 16, tN, 16, tN>;
    static_assert(tileATail::LogicalTileBytes + tileB::LogicalTileBytes <=
                      256 * 1024, "SharedTReg capacity exceeded");
    static_assert(2 * cFull::LogicalTileBytes + cTail::LogicalTileBytes +
                      2 * tileA::LogicalTileBytes <= 256 * 1024,
                  "CellReg capacity exceeded");

    const int tid = static_cast<int>(get_thread_idx());
    constexpr int kFullK = gK / tK;
    for (int n = 0; n < gN / tN; ++n) {
        const int n_begin = n * tN;
        cFull c0;
        cFull c1;
        cTail c_tail;
        matmul_reuse_b_mtail_shared_a_detail::run_k<
            true, false, tK, dtype, gM, gN, gK, tN, tK>(
                a_ptr, b_ptr, c0, c1, c_tail, n_begin, 0, tid);
#pragma clang loop unroll(disable)
        for (int k = 1; k < kFullK; ++k) {
            matmul_reuse_b_mtail_shared_a_detail::run_k<
                false, false, tK, dtype, gM, gN, gK, tN, tK>(
                    a_ptr, b_ptr, c0, c1, c_tail, n_begin, k * tK, tid);
        }
        matmul_reuse_b_mtail_shared_a_detail::run_k_tail<
            96, kFullK * tK, false, dtype, gM, gN, gK, tN, tK>(
                a_ptr, b_ptr, c0, c1, c_tail, n_begin, tid);

        using gmC = global_tensor<float, RowMajor<gM, gN>>;
        gmC gc0(c_ptr + tid * 32 * gN + n_begin);
        gmC gc1(c_ptr + (128 + tid * 32) * gN + n_begin);
        TSTORE_CUBE(gc0, c0);
        TSTORE_CUBE(gc1, c1);
        if (tid < 2) {
            gmC gc_tail(c_ptr + (256 + tid * 16) * gN + n_begin);
            TSTORE_CUBE(gc_tail, c_tail);
        }
    }
}
