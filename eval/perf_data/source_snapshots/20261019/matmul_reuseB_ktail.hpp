#pragma once

#include "matmul_reuseB.hpp"

// Experimental reuse-B variant for logical K extents that do not divide tK.
// Complete tK slices keep B resident across M strips; the remaining K is
// decomposed into supported power-of-two slices (7392 % 128 = 64 + 32).
namespace matmul_reuse_b_ktail_detail {

template <int ChunkK, bool FirstK, bool LastK, int ResidentM,
          typename dtype, int gM, int gN, int gK, int tN, int... Ms>
PTO_SHARED_INLINE void run_full_m_chunk(
    dtype *a_ptr, dtype *b_ptr,
    MatmulReuseAStorage<CubeAccumulatorM32<float, 32, tN>, ResidentM> &cs,
    int m_base, int n_block, int k_begin, int tid,
    std::integer_sequence<int, Ms...>) {
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA = CubeTileM32<dtype, 32, ChunkK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, ChunkK>;
    using iterA = global_iterator<gmA, tileA>;
    using iterB = global_iterator<gmB, tileBMatrix>;
    constexpr auto options = fixp::keep_acc();

    iterA as(a_ptr);
    iterB bs(b_ptr);
    SharedTile<tileBMatrix> b;
    auto gb = bs(n_block, k_begin / ChunkK);
    TLOAD<tileBMatrix, 1>(b, gb);
    ([&] {
        tileA a;
        auto ga = as((m_base + Ms * 128) / 32 + tid, k_begin / ChunkK);
        TLOAD_CUBE(a, ga);
        if constexpr (FirstK) {
            if constexpr (LastK) {
                TMATMUL(cs[Ms], a, b, options, 128);
            } else {
                TMATMUL(cs[Ms], a, b, options.raw_acc(), 128);
            }
        } else if constexpr (LastK) {
            TMATMUL_ACC(cs[Ms], cs[Ms], a, b,
                        options.acc_hint(), 128);
        } else {
            TMATMUL_ACC(cs[Ms], cs[Ms], a, b,
                        options.raw_acc().acc_hint(), 128);
        }
    }(), ...);
}

template <int RemainingK, int OffsetK, int ResidentM,
          typename dtype, int gM, int gN, int gK, int tN, int... Ms>
PTO_SHARED_INLINE void run_full_m_tail(
    dtype *a_ptr, dtype *b_ptr,
    MatmulReuseAStorage<CubeAccumulatorM32<float, 32, tN>, ResidentM> &cs,
    int m_base, int n_block, int tid, std::integer_sequence<int, Ms...> seq) {
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk % 32 == 0,
                  "K tail needs supported power-of-two chunks >= 32");
    run_full_m_chunk<chunk, false, RemainingK == chunk, ResidentM,
                     dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, cs, m_base, n_block, OffsetK, tid, seq);
    if constexpr (RemainingK > chunk) {
        run_full_m_tail<RemainingK - chunk, OffsetK + chunk, ResidentM,
                        dtype, gM, gN, gK, tN>(
            a_ptr, b_ptr, cs, m_base, n_block, tid, seq);
    }
}

template <int ResidentM, typename dtype, int gM, int gN, int gK,
          int tN, int tK>
PTO_SHARED_INLINE void run_full_m_group(float *c_ptr, dtype *a_ptr,
                                         dtype *b_ptr, int m_base,
                                         int n_block) {
    constexpr int full_k = gK / tK;
    constexpr int tail_k = gK % tK;
    using tileC = CubeAccumulatorM32<float, 32, tN>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using iterC = global_iterator<gmC, tileC>;
    using seq = std::make_integer_sequence<int, ResidentM>;
    static_assert(full_k > 0 && tail_k > 0);
    static_assert(ResidentM * tileC::LogicalTileBytes +
                      CubeTileM32<dtype, 32, tK>::LogicalTileBytes <=
                      256 * 1024, "CellReg A + resident C exceeds capacity");
    static_assert(SharedMatrixRight<dtype, tN, tK>::LogicalTileBytes <=
                      256 * 1024, "SharedTile B exceeds capacity");

    int tid = static_cast<int>(get_thread_idx());
    MatmulReuseAStorage<tileC, ResidentM> cs;
    run_full_m_chunk<tK, true, false, ResidentM, dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, cs, m_base, n_block, 0, tid, seq{});
#pragma clang loop unroll(disable)
    for (int k = 1; k < full_k; ++k) {
        run_full_m_chunk<tK, false, false, ResidentM, dtype, gM, gN, gK, tN>(
            a_ptr, b_ptr, cs, m_base, n_block, k * tK, tid, seq{});
    }
    run_full_m_tail<tail_k, full_k * tK, ResidentM,
                    dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, cs, m_base, n_block, tid, seq{});

    iterC outputs(c_ptr);
#pragma clang loop unroll(full)
    for (int mi = 0; mi < ResidentM; ++mi) {
        auto gc = outputs((m_base + mi * 128) / 32 + tid, n_block);
        TSTORE_CUBE(gc, cs[mi]);
    }
}

template <int ChunkK, bool FirstK, bool LastK,
          typename dtype, int gM, int gN, int gK, int tN>
PTO_SHARED_INLINE void run_short_m_chunk(
    dtype *a_ptr, dtype *b_ptr,
    CubeAccumulatorM16<float, 16, tN, 16, tN> &c,
    int m_base, int n_begin, int k_begin, int tid) {
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA = CubeTileM16<dtype, 16, ChunkK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, ChunkK>;
    constexpr auto options = fixp::keep_acc();
    tileA a;
    SharedTile<tileBMatrix> b;
    const int local_m = tid < 2 ? m_base + tid * 16 : m_base;
    gmA ga(a_ptr + local_m * gK + k_begin);
    gmB gb(b_ptr + n_begin * gK + k_begin);
    TLOAD_CUBE(a, ga);
    TLOAD<tileBMatrix, 1>(b, gb);
    if constexpr (FirstK) {
        if constexpr (LastK) {
            TMATMUL(c, a, b, options, 32);
        } else {
            TMATMUL(c, a, b, options.raw_acc(), 32);
        }
    } else if constexpr (LastK) {
        TMATMUL_ACC(c, c, a, b, options.acc_hint(), 32);
    } else {
        TMATMUL_ACC(c, c, a, b, options.raw_acc().acc_hint(), 32);
    }
}

template <int RemainingK, int OffsetK,
          typename dtype, int gM, int gN, int gK, int tN>
PTO_SHARED_INLINE void run_short_m_tail(
    dtype *a_ptr, dtype *b_ptr,
    CubeAccumulatorM16<float, 16, tN, 16, tN> &c,
    int m_base, int n_begin, int tid) {
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk % 32 == 0,
                  "K tail needs supported power-of-two chunks >= 32");
    run_short_m_chunk<chunk, false, RemainingK == chunk,
                      dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, c, m_base, n_begin, OffsetK, tid);
    if constexpr (RemainingK > chunk) {
        run_short_m_tail<RemainingK - chunk, OffsetK + chunk,
                         dtype, gM, gN, gK, tN>(
            a_ptr, b_ptr, c, m_base, n_begin, tid);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK>
PTO_SHARED_INLINE void run_short_m_group(float *c_ptr, dtype *a_ptr,
                                          dtype *b_ptr, int m_base,
                                          int n_begin) {
    constexpr int full_k = gK / tK;
    constexpr int tail_k = gK % tK;
    using tileC = CubeAccumulatorM16<float, 16, tN, 16, tN>;
    static_assert(full_k > 0 && tail_k > 0);
    const int tid = static_cast<int>(get_thread_idx());
    tileC c;
    run_short_m_chunk<tK, true, false, dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, c, m_base, n_begin, 0, tid);
#pragma clang loop unroll(disable)
    for (int k = 1; k < full_k; ++k) {
        run_short_m_chunk<tK, false, false, dtype, gM, gN, gK, tN>(
            a_ptr, b_ptr, c, m_base, n_begin, k * tK, tid);
    }
    run_short_m_tail<tail_k, full_k * tK, dtype, gM, gN, gK, tN>(
        a_ptr, b_ptr, c, m_base, n_begin, tid);
    if (tid < 2) {
        using gmC = global_tensor<float, RowMajor<gM, gN>>;
        gmC gc(c_ptr + (m_base + tid * 16) * gN + n_begin);
        TSTORE_CUBE(gc, c);
    }
}

}  // namespace matmul_reuse_b_ktail_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseB_ktail(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    static_assert(tM == 128 && gM >= 128 && gM % 128 == 32,
                  "experimental K-tail path expects a 32-row M tail");
    static_assert(gN % tN == 0 && (tN & (tN - 1)) == 0,
                  "experimental K-tail path requires full power-of-two N tiles");
    static_assert(gK % tK != 0 && tK >= 64 && (tK & (tK - 1)) == 0,
                  "experimental K-tail path requires a power-of-two full K tile");
    constexpr int full_m = gM / 128;
    constexpr int resident_m = full_m < 4 ? full_m : 4;
    constexpr int m_groups = full_m / resident_m;
    constexpr int remaining_full_m = full_m % resident_m;
#pragma clang loop unroll(disable)
    for (int n = 0; n < gN / tN; ++n) {
#pragma clang loop unroll(disable)
        for (int group = 0; group < m_groups; ++group) {
            matmul_reuse_b_ktail_detail::run_full_m_group<
                resident_m, dtype, gM, gN, gK, tN, tK>(
                c_ptr, a_ptr, b_ptr, group * resident_m * 128, n);
        }
        if constexpr (remaining_full_m > 0) {
            matmul_reuse_b_ktail_detail::run_full_m_group<
                remaining_full_m, dtype, gM, gN, gK, tN, tK>(
                c_ptr, a_ptr, b_ptr, m_groups * resident_m * 128, n);
        }
        matmul_reuse_b_ktail_detail::run_short_m_group<
            dtype, gM, gN, gK, tN, tK>(
            c_ptr, a_ptr, b_ptr, full_m * 128, n * tN);
    }
}
