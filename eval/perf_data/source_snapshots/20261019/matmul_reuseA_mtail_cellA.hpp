#pragma once

#include "matmul_reuseA.hpp"

// ReuseA test for M=128+128+32, K tail=96. A stays PE-local and is reused
// across N tiles. Each M strip has its own K chain, as in the original reuseA.
namespace matmul_reuse_a_mtail_cell_a_detail {

template <int ValidM, int NCount, bool FirstK, bool LastK, int ValidK,
          typename dtype, int gM, int gN, int gK, int tN, int tK,
          typename TileC>
PTO_SHARED_INLINE void run_k(
    dtype *a_ptr, dtype *b_ptr, MatmulReuseAStorage<TileC, NCount> &cs,
    int m_begin, int n_start, int k_begin, int tid) {
    constexpr int PeM = ValidM == 32 ? 16 : 32;
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA16 = CubeTileM16<dtype, PeM, tK, PeM, ValidK>;
    using tileA32 = CubeTileM32<dtype, PeM, tK, PeM, ValidK>;
    using tileA = std::conditional_t<(ValidM == 32), tileA16, tileA32>;
    using tileB = SharedMatrixRight<dtype, tN, tK, tN, ValidK>;
    constexpr auto options = fixp::keep_acc();

    const int row = tid * PeM < ValidM ? m_begin + tid * PeM : m_begin;
    tileA a;
    gmA ga(a_ptr + row * gK + k_begin);
    TLOAD_CUBE(a, ga);
#pragma clang loop unroll(full)
    for (int n = 0; n < NCount; ++n) {
        SharedTile<tileB> b;
        gmB gb(b_ptr + (n_start + n) * tN * gK + k_begin);
        TLOAD<tileB, 1>(b, gb);
        if constexpr (FirstK) {
            if constexpr (LastK) {
                TMATMUL(cs[n], a, b, options, ValidM);
            } else {
                TMATMUL(cs[n], a, b, options.raw_acc(), ValidM);
            }
        } else if constexpr (LastK) {
            TMATMUL_ACC(cs[n], cs[n], a, b, options.acc_hint(), ValidM);
        } else {
            TMATMUL_ACC(cs[n], cs[n], a, b,
                        options.raw_acc().acc_hint(), ValidM);
        }
    }
}

template <int ValidM, int NCount, int RemainingK, int OffsetK, bool FirstK,
          typename dtype, int gM, int gN, int gK, int tN, int tK,
          typename TileC>
PTO_SHARED_INLINE void run_k_tail(
    dtype *a_ptr, dtype *b_ptr, MatmulReuseAStorage<TileC, NCount> &cs,
    int m_begin, int n_start, int tid) {
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk <= tK);
    run_k<ValidM, NCount, FirstK, RemainingK == chunk, chunk,
          dtype, gM, gN, gK, tN, tK>(
        a_ptr, b_ptr, cs, m_begin, n_start, OffsetK, tid);
    if constexpr (RemainingK > chunk) {
        run_k_tail<ValidM, NCount, RemainingK - chunk, OffsetK + chunk,
                   false, dtype, gM, gN, gK, tN, tK>(
            a_ptr, b_ptr, cs, m_begin, n_start, tid);
    }
}

template <int ValidM, int NCount, typename dtype,
          int gM, int gN, int gK, int tN, int tK>
__attribute__((noinline)) void run_strip(
    float *c_ptr, dtype *a_ptr, dtype *b_ptr, int m_begin, int n_start) {
    constexpr int PeM = ValidM == 32 ? 16 : 32;
    using tileA16 = CubeTileM16<dtype, PeM, tK>;
    using tileA32 = CubeTileM32<dtype, PeM, tK>;
    using tileA = std::conditional_t<(ValidM == 32), tileA16, tileA32>;
    using c16 = CubeAccumulatorM16<float, PeM, tN, PeM, tN>;
    using c32 = CubeAccumulatorM32<float, PeM, tN, PeM, tN>;
    using tileC = std::conditional_t<(ValidM == 32), c16, c32>;
    using tileB = SharedMatrixRight<dtype, tN, tK>;
    static_assert(tileB::LogicalTileBytes <= 256 * 1024);
    static_assert(tileA::LogicalTileBytes +
                      NCount * tileC::LogicalTileBytes <= 256 * 1024);

    MatmulReuseAStorage<tileC, NCount> cs;
    const int tid = static_cast<int>(get_thread_idx());
    constexpr int kFullK = gK / tK;
    run_k<ValidM, NCount, true, false, tK,
          dtype, gM, gN, gK, tN, tK>(
        a_ptr, b_ptr, cs, m_begin, n_start, 0, tid);
#pragma clang loop unroll(disable)
    for (int k = 1; k < kFullK; ++k) {
        run_k<ValidM, NCount, false, false, tK,
              dtype, gM, gN, gK, tN, tK>(
            a_ptr, b_ptr, cs, m_begin, n_start, k * tK, tid);
    }
    run_k_tail<ValidM, NCount, 96, kFullK * tK, false,
               dtype, gM, gN, gK, tN, tK>(
        a_ptr, b_ptr, cs, m_begin, n_start, tid);

    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    if (tid * PeM < ValidM) {
#pragma clang loop unroll(full)
        for (int n = 0; n < NCount; ++n) {
            gmC gc(c_ptr + (m_begin + tid * PeM) * gN +
                   (n_start + n) * tN);
            TSTORE_CUBE(gc, cs[n]);
        }
    }
}

}  // namespace matmul_reuse_a_mtail_cell_a_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseA_mtail_cellA(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    static_assert(gM == 288 && tM == 128, "test variant requires M288/tM128");
    static_assert(gN % tN == 0 && gN / tN <= 2,
                  "test variant requires one or two full N tiles");
    static_assert(gK > tK && gK % tK == 96,
                  "test variant requires a 96-column K tail");
    static_assert(tK >= 128 && (tK & (tK - 1)) == 0);
    constexpr int NBlocks = gN / tN;
    matmul_reuse_a_mtail_cell_a_detail::run_strip<
        128, NBlocks, dtype, gM, gN, gK, tN, tK>(
            c_ptr, a_ptr, b_ptr, 0, 0);
    matmul_reuse_a_mtail_cell_a_detail::run_strip<
        128, NBlocks, dtype, gM, gN, gK, tN, tK>(
            c_ptr, a_ptr, b_ptr, 128, 0);
#pragma clang loop unroll(disable)
    for (int n = 0; n < NBlocks; ++n) {
        matmul_reuse_a_mtail_cell_a_detail::run_strip<
            32, 1, dtype, gM, gN, gK, tN, tK>(
                c_ptr, a_ptr, b_ptr, 256, n);
    }
}
