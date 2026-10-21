#pragma once

#include "matmul_reuseA.hpp"
#include <utility>

namespace matmul_reuse_b_detail {

template <bool FirstK, bool LastK, int ValidK, int ResidentM,
          typename dtype, int gM, int gN, int gK, int tN, int tK, int... Ms>
PTO_SHARED_INLINE void run_k_tail(
    dtype *a_ptr, dtype *b_ptr,
    MatmulReuseAStorage<CubeAccumulatorM32<float, 32, tN>, ResidentM> &cs,
    int m_base, int n_block, int k_block, int tid,
    std::integer_sequence<int, Ms...>) {
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA = CubeTileM32<dtype, 32, tK, 32, ValidK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK, tN, ValidK>;
    constexpr auto options = fixp::keep_acc();

    SharedTile<tileBMatrix> b;
    gmB gb(b_ptr + n_block * tN * gK + k_block);
    TLOAD<tileBMatrix, 1>(b, gb);
    ([&] {
        tileA a;
        gmA ga(a_ptr + (m_base + Ms * 128 + tid * 32) * gK + k_block);
        TLOAD_CUBE(a, ga);
        if constexpr (FirstK) {
            if constexpr (LastK) {
                TMATMUL(cs[Ms], a, b, options, 128);
            } else {
                TMATMUL(cs[Ms], a, b, options.raw_acc(), 128);
            }
        } else if constexpr (LastK) {
            TMATMUL_ACC(cs[Ms], cs[Ms], a, b, options.acc_hint(), 128);
        } else {
            TMATMUL_ACC(cs[Ms], cs[Ms], a, b,
                        options.raw_acc().acc_hint(), 128);
        }
    }(), ...);
}

template <int RemainingK, int OffsetK, bool FirstK, int ResidentM,
          typename dtype, int gM, int gN, int gK, int tN, int tK,
          int... Ms>
PTO_SHARED_INLINE void run_k_tail_chain(
    dtype *a_ptr, dtype *b_ptr,
    MatmulReuseAStorage<CubeAccumulatorM32<float, 32, tN>, ResidentM> &cs,
    int m_base, int n_block, int tid, std::integer_sequence<int, Ms...> seq) {
    // CUBE requires a power-of-two effective K: keep the physical tK tile
    // and express each tail chunk through A/B's valid_col (e.g. 96=64+32).
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk <= tK);
    run_k_tail<FirstK, RemainingK == chunk, chunk, ResidentM,
               dtype, gM, gN, gK, tN, tK>(
        a_ptr, b_ptr, cs, m_base, n_block, OffsetK, tid, seq);
    if constexpr (RemainingK > chunk) {
        run_k_tail_chain<RemainingK - chunk, OffsetK + chunk, false,
                         ResidentM, dtype, gM, gN, gK, tN, tK>(
            a_ptr, b_ptr, cs, m_base, n_block, tid, seq);
    }
}

template <bool FirstK, bool LastK, int ResidentM, typename dtype,
          int gM, int gN, int gK, int tN, int tK, int... Ms>
PTO_SHARED_INLINE void run_k(
    global_iterator<global_tensor<dtype, RowMajor<gM, gK>>,
                    CubeTileM32<dtype, 32, tK>> &gIterA,
    global_iterator<global_tensor<dtype, RowMajor<gN, gK>>,
                    SharedMatrixRight<dtype, tN, tK>> &gIterB,
    MatmulReuseAStorage<CubeAccumulatorM32<float, 32, tN>, ResidentM> &tCGroup,
    int mBase, int nBlock, int kBlock, int tid,
    std::integer_sequence<int, Ms...>) {
    using tileA = CubeTileM32<dtype, 32, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileB = SharedTile<tileBMatrix>;
    constexpr auto options = fixp::keep_acc();

    // One global B load is consumed by every M strip in this resident group.
    tileB tB;
    auto gB = gIterB(nBlock, kBlock);
    TLOAD<tileBMatrix, 1>(tB, gB);
    ([&] {
        tileA tA;
        auto gA = gIterA((mBase + Ms * 128) / 32 + tid, kBlock);
        TLOAD_CUBE(tA, gA);
        if constexpr (FirstK) {
            if constexpr (LastK) {
                TMATMUL(tCGroup[Ms], tA, tB, options, 128);
            } else {
                TMATMUL(tCGroup[Ms], tA, tB, options.raw_acc(), 128);
            }
        } else if constexpr (LastK) {
            TMATMUL_ACC(tCGroup[Ms], tCGroup[Ms], tA, tB,
                        options.acc_hint(), 128);
        } else {
            TMATMUL_ACC(tCGroup[Ms], tCGroup[Ms], tA, tB,
                        options.raw_acc().acc_hint(), 128);
        }
    }(), ...);
}

template <int ResidentM, typename dtype, int gM, int gN, int gK,
          int tN, int tK>
PTO_SHARED_INLINE void run_m_group(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                   int mBase, int nBlock) {
    constexpr int kKb = gK / tK;
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using tileA = CubeTileM32<dtype, 32, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileC = CubeAccumulatorM32<float, 32, tN>;
    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;
    constexpr int kTailK = gK % tK;
    static_assert(kKb > 0 || kTailK > 0);
    static_assert(ResidentM * tileC::LogicalTileBytes +
                      tileA::LogicalTileBytes <= 256 * 1024,
                  "CellReg cannot hold A and resident C accumulators");
    static_assert(tileBMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tile exceeds SharedTReg capacity");

    const int tid = static_cast<int>(get_thread_idx());
    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);
    MatmulReuseAStorage<tileC, ResidentM> tCGroup;
    using MIndices = std::make_integer_sequence<int, ResidentM>;

    if constexpr (kKb > 0) {
        run_k<true, kKb == 1 && kTailK == 0, ResidentM,
              dtype, gM, gN, gK, tN, tK>(
            gIterA, gIterB, tCGroup, mBase, nBlock, 0, tid, MIndices{});
    }
    if constexpr (kKb > 2) {
#pragma clang loop unroll(disable)
        for (int k = 1; k < kKb - 1; ++k) {
            run_k<false, false, ResidentM, dtype, gM, gN, gK, tN, tK>(
                gIterA, gIterB, tCGroup, mBase, nBlock, k, tid, MIndices{});
        }
    }
    if constexpr (kKb > 1) {
        run_k<false, kTailK == 0, ResidentM, dtype, gM, gN, gK, tN, tK>(
            gIterA, gIterB, tCGroup, mBase, nBlock, kKb - 1, tid,
            MIndices{});
    }
    if constexpr (kTailK > 0) {
        run_k_tail_chain<kTailK, kKb * tK, kKb == 0, ResidentM,
                         dtype, gM, gN, gK, tN, tK>(
            a_ptr, b_ptr, tCGroup, mBase, nBlock, tid, MIndices{});
    }

#pragma clang loop unroll(full)
    for (int mi = 0; mi < ResidentM; ++mi) {
        auto gC = gIterC((mBase + mi * 128) / 32 + tid, nBlock);
        TSTORE_CUBE(gC, tCGroup[mi]);
    }
}

template <bool FirstK, bool LastK, int ValidM, int ValidN, int ValidK,
          typename dtype, int gM, int gN, int gK, int tN, int tK,
          typename TileC>
PTO_SHARED_INLINE void run_edge_k(
    TileC &c, dtype *a_ptr, dtype *b_ptr,
    int m_begin, int n_begin, int k_begin, int tid) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using tileA16 = CubeTileM16<dtype, kPeM, tK, kPeM, ValidK>;
    using tileA32 = CubeTileM32<dtype, kPeM, tK, kPeM, ValidK>;
    using tileA = std::conditional_t<(kPeM <= 16), tileA16, tileA32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK, ValidN, ValidK>;
    constexpr auto options = fixp::keep_acc();

    const int local_m = tid * kPeM < ValidM
        ? m_begin + tid * kPeM : m_begin;
    tileA a;
    SharedTile<tileBMatrix> b;
    gmA ga(a_ptr + local_m * gK + k_begin);
    gmB gb(b_ptr + n_begin * gK + k_begin);
    TLOAD_CUBE(a, ga);
    TLOAD<tileBMatrix, 1>(b, gb);
    if constexpr (FirstK) {
        if constexpr (LastK) {
            TMATMUL(c, a, b, options, ValidM);
        } else {
            TMATMUL(c, a, b, options.raw_acc(), ValidM);
        }
    } else if constexpr (LastK) {
        TMATMUL_ACC(c, c, a, b, options.acc_hint(), ValidM);
    } else {
        TMATMUL_ACC(c, c, a, b, options.raw_acc().acc_hint(), ValidM);
    }
}

template <int RemainingK, int OffsetK, bool FirstK, int ValidM,
          int ValidN, typename dtype, int gM, int gN, int gK,
          int tN, int tK, typename TileC>
PTO_SHARED_INLINE void run_edge_k_tail_chain(
    TileC &c, dtype *a_ptr, dtype *b_ptr,
    int m_begin, int n_begin, int tid) {
    constexpr int chunk = matmul_reuse_a_detail::floor_power_of_two(RemainingK);
    static_assert(chunk >= 32 && chunk <= tK);
    run_edge_k<FirstK, RemainingK == chunk, ValidM, ValidN, chunk,
               dtype, gM, gN, gK, tN, tK>(
        c, a_ptr, b_ptr, m_begin, n_begin, OffsetK, tid);
    if constexpr (RemainingK > chunk) {
        run_edge_k_tail_chain<RemainingK - chunk, OffsetK + chunk,
                              false, ValidM, ValidN, dtype,
                              gM, gN, gK, tN, tK>(
            c, a_ptr, b_ptr, m_begin, n_begin, tid);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int ValidN>
PTO_SHARED_INLINE void run_edge_block(float *c_ptr, dtype *a_ptr,
                                      dtype *b_ptr, int m_begin,
                                      int n_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int kFullK = gK / tK;
    constexpr int kTailK = gK % tK;
    using tileC16 = CubeAccumulatorM16<float, kPeM, tN, kPeM, ValidN>;
    using tileC32 = CubeAccumulatorM32<float, kPeM, tN, kPeM, ValidN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileC16, tileC32>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    const int tid = static_cast<int>(get_thread_idx());
    tileC c;
    if constexpr (kFullK > 0) {
        run_edge_k<true, kFullK == 1 && kTailK == 0,
                   ValidM, ValidN, tK, dtype, gM, gN, gK, tN, tK>(
            c, a_ptr, b_ptr, m_begin, n_begin, 0, tid);
    }
    if constexpr (kFullK > 2) {
#pragma clang loop unroll(disable)
        for (int k = 1; k < kFullK - 1; ++k) {
            run_edge_k<false, false, ValidM, ValidN, tK,
                       dtype, gM, gN, gK, tN, tK>(
                c, a_ptr, b_ptr, m_begin, n_begin, k * tK, tid);
        }
    }
    if constexpr (kFullK > 1) {
        run_edge_k<false, kTailK == 0, ValidM, ValidN, tK,
                   dtype, gM, gN, gK, tN, tK>(
            c, a_ptr, b_ptr, m_begin, n_begin, (kFullK - 1) * tK, tid);
    }
    if constexpr (kTailK > 0) {
        run_edge_k_tail_chain<kTailK, kFullK * tK, kFullK == 0,
                              ValidM, ValidN, dtype, gM, gN, gK, tN, tK>(
            c, a_ptr, b_ptr, m_begin, n_begin, tid);
    }
    if (tid * kPeM < ValidM) {
        gmC gc(c_ptr + (m_begin + tid * kPeM) * gN + n_begin);
        TSTORE_CUBE(gc, c);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int RemainingN>
PTO_SHARED_INLINE void run_edge_n_tail(float *c_ptr, dtype *a_ptr,
                                       dtype *b_ptr, int m_begin,
                                       int n_begin) {
    constexpr int kChunkN = matmul_reuse_a_detail::floor_power_of_two(RemainingN);
    run_edge_block<dtype, gM, gN, gK, tN, tK, ValidM, kChunkN>(
        c_ptr, a_ptr, b_ptr, m_begin, n_begin);
    if constexpr (RemainingN > kChunkN) {
        run_edge_n_tail<dtype, gM, gN, gK, tN, tK,
                        ValidM, RemainingN - kChunkN>(
            c_ptr, a_ptr, b_ptr, m_begin, n_begin + kChunkN);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_edge_m_strip(float *c_ptr, dtype *a_ptr,
                                        dtype *b_ptr, int m_begin) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < gN / tN; ++n) {
        run_edge_block<dtype, gM, gN, gK, tN, tK, ValidM, tN>(
            c_ptr, a_ptr, b_ptr, m_begin, n * tN);
    }
    if constexpr (gN % tN > 0) {
        run_edge_n_tail<dtype, gM, gN, gK, tN, tK,
                        ValidM, gN % tN>(
            c_ptr, a_ptr, b_ptr, m_begin, (gN / tN) * tN);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int RemainingM>
PTO_SHARED_INLINE void run_edge_m_tail(float *c_ptr, dtype *a_ptr,
                                       dtype *b_ptr, int m_begin) {
    constexpr int kChunkM = matmul_reuse_a_detail::floor_power_of_two(RemainingM);
    run_edge_m_strip<dtype, gM, gN, gK, tN, tK, kChunkM>(
        c_ptr, a_ptr, b_ptr, m_begin);
    if constexpr (RemainingM > kChunkM) {
        run_edge_m_tail<dtype, gM, gN, gK, tN, tK,
                        RemainingM - kChunkM>(
            c_ptr, a_ptr, b_ptr, m_begin + kChunkM);
    }
}

}  // namespace matmul_reuse_b_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseB(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kFullMBlocks = gM / kGroupM;
    constexpr int kTailM = gM % kGroupM;
    constexpr int kFullNBlocks = gN / tN;
    constexpr int kTailN = gN % tN;
    static_assert(gK > 0 && tK > 0);
    static_assert(gK % tK == 0 ||
                      (gK % tK >= 32 && (gK % tK) % 32 == 0),
                  "K tail must be a multiple of 32 and at least 32");
    static_assert(gK % tK == 0 ||
                      (tK >= 64 && (tK & (tK - 1)) == 0),
                  "K tail requires a power-of-two physical tK >= 64");
    static_assert(tM % 4 == 0 && tM % kGroupM == 0);
    static_assert((kGroupM & (kGroupM - 1)) == 0);
    static_assert((tN & (tN - 1)) == 0);

    if constexpr (kGroupM == 128 && kFullMBlocks > 0) {
        constexpr int kResidentM = kFullMBlocks < 4 ? kFullMBlocks : 4;
        constexpr int kMGroups = kFullMBlocks / kResidentM;
        constexpr int kMTail = kFullMBlocks % kResidentM;
#pragma clang loop unroll(disable)
        for (int n = 0; n < kFullNBlocks; ++n) {
#pragma clang loop unroll(disable)
            for (int group = 0; group < kMGroups; ++group) {
                matmul_reuse_b_detail::run_m_group<
                    kResidentM, dtype, gM, gN, gK, tN, tK>(
                    c_ptr, a_ptr, b_ptr, group * kResidentM * 128, n);
            }
            if constexpr (kMTail > 0) {
                matmul_reuse_b_detail::run_m_group<
                    kMTail, dtype, gM, gN, gK, tN, tK>(
                    c_ptr, a_ptr, b_ptr, kMGroups * kResidentM * 128, n);
            }
        }
        if constexpr (kTailN > 0) {
#pragma clang loop unroll(full)
            for (int m = 0; m < kFullMBlocks; ++m) {
                if constexpr (gK % tK == 0) {
                    matmul_reuse_a_detail::run_n_tail<
                        dtype, gM, gN, gK, tN, tK, 128, kTailN>(
                        c_ptr, a_ptr, b_ptr, m * 128, kFullNBlocks * tN);
                } else {
                    matmul_reuse_b_detail::run_edge_n_tail<
                        dtype, gM, gN, gK, tN, tK, 128, kTailN>(
                        c_ptr, a_ptr, b_ptr, m * 128, kFullNBlocks * tN);
                }
            }
        }
        if constexpr (kTailM > 0) {
            if constexpr (gK % tK == 0) {
                matmul_reuse_a_detail::run_m_tail<
                    dtype, gM, gN, gK, tN, tK, kTailM>(
                    c_ptr, a_ptr, b_ptr, kFullMBlocks * 128);
            } else {
                matmul_reuse_b_detail::run_edge_m_tail<
                    dtype, gM, gN, gK, tN, tK, kTailM>(
                    c_ptr, a_ptr, b_ptr, kFullMBlocks * 128);
            }
        }
    } else {
        if constexpr (gK % tK == 0) {
            matmul_reuseA<dtype, gM, gN, gK, tM, tN, tK>(
                c_ptr, a_ptr, b_ptr);
        } else {
#pragma clang loop unroll(disable)
            for (int m = 0; m < gM / 128; ++m) {
                matmul_reuse_b_detail::run_edge_m_strip<
                    dtype, gM, gN, gK, tN, tK, 128>(
                    c_ptr, a_ptr, b_ptr, m * 128);
            }
            if constexpr (gM % 128 > 0) {
                matmul_reuse_b_detail::run_edge_m_tail<
                    dtype, gM, gN, gK, tN, tK, gM % 128>(
                    c_ptr, a_ptr, b_ptr, (gM / 128) * 128);
            }
        }
    }
}
