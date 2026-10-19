#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

template <typename TileT, int Count>
struct MatmulReuseAStorage {
    TileT tiles[Count];
    TileT &operator[](int index) { return tiles[index]; }
};

namespace matmul_reuse_a_detail {

constexpr int floor_power_of_two(int value) {
    int result = 1;
    while ((result << 1) <= value) result <<= 1;
    return result;
}

constexpr int compute_group_m(int valid_m) {
    // Current model requires power-of-two TMATMUL M.
    int rows = 1;
    while (rows < valid_m) rows <<= 1;
    return rows;
}

template <int ValidM, int PeM>
PTO_SHARED_INLINE int local_m_offset(int tid) {
    if constexpr (ValidM > 64 && ValidM < 128) {
        // Model requires full 32-row Local-A descriptors even when a load
        // zero-pads its CELL. Shift the last shard inside A instead. E.g.
        // M120 starts are 0,32,64,88; stores partition 32,32,24,32 rows.
        return tid * PeM < ValidM - PeM ? tid * PeM : ValidM - PeM;
    }
    return tid * PeM < ValidM ? tid * PeM : 0;
}

template <int ValidM, typename TileC, typename GmC>
PTO_SHARED_INLINE void store_c(GmC &gc, TileC &c, int tid) {
    constexpr int PeM = TileC::Rows;
    if constexpr (ValidM > 64 && ValidM < 128) {
        using StoreC = CubeAccumulatorM32<float, PeM, TileC::Cols,
                                         -1, TileC::ValidCol>;
        const int start = local_m_offset<ValidM, PeM>(tid);
        const int end = tid == 3 ? ValidM : local_m_offset<ValidM, PeM>(tid + 1);
        if (end > start) {
            StoreC stored(end - start);
            stored.data() = c.data();
            TSTORE_CUBE(gc, stored);
        }
    } else if constexpr (ValidM >= 4 * PeM) {
        TSTORE_CUBE(gc, c);
    } else if (tid * PeM < ValidM) {
        using StoreC = std::conditional_t<PeM == 16,
            CubeAccumulatorM16<float, PeM, TileC::Cols, -1, TileC::ValidCol>,
            CubeAccumulatorM32<float, PeM, TileC::Cols, -1, TileC::ValidCol>>;
        const int remaining = ValidM - tid * PeM;
        StoreC stored(remaining < PeM ? remaining : PeM);
        stored.data() = c.data();
        TSTORE_CUBE(gc, stored);
    }
}

template <bool FirstK, bool LastK, int ReuseN, typename dtype,
          int gM, int gN, int gK, int tN, int tK, int ValidM,
          typename TileC>
PTO_SHARED_INLINE void run_full_n_k(
    dtype *a_ptr, dtype *b_ptr, MatmulReuseAStorage<TileC, ReuseN> &cs,
    int m_begin, int j_base, int k, int tid) {
    constexpr int PeM = ValidM <= 64 ? 16 : 32;
    constexpr int ComputeM = compute_group_m(ValidM);
    using TileA = std::conditional_t<PeM == 16,
        CubeTileM16<dtype, PeM, tK>, CubeTileM32<dtype, PeM, tK>>;
    using BMatrix = SharedMatrixRight<dtype, tN, tK>;
    using GmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using GmB = global_tensor<dtype, RowMajor<gN, gK>>;
    constexpr auto options = fixp::keep_acc();
    TileA a;
    GmA ga(a_ptr + (m_begin + local_m_offset<ValidM, PeM>(tid)) * gK + k * tK);
    TLOAD_CUBE(a, ga);
#pragma clang loop unroll(full)
    for (int n = 0; n < ReuseN; ++n) {
        if (j_base + n < gN / tN) {
            SharedTile<BMatrix> b;
            GmB gb(b_ptr + (j_base + n) * tN * gK + k * tK);
            TLOAD<BMatrix, 1>(b, gb);
            if constexpr (FirstK && LastK) {
                TMATMUL(cs[n], a, b, options, ComputeM);
            } else if constexpr (FirstK) {
                TMATMUL(cs[n], a, b, options.raw_acc(), ComputeM);
            } else if constexpr (LastK) {
                TMATMUL_ACC(cs[n], cs[n], a, b, options.acc_hint(), ComputeM);
            } else {
                TMATMUL_ACC(cs[n], cs[n], a, b,
                            options.raw_acc().acc_hint(), ComputeM);
            }
        }
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_full_n(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using tileAM16 = CubeTileM16<dtype, kPeM, tK>;
    using tileAM32 = CubeTileM32<dtype, kPeM, tK>;
    using tileA = std::conditional_t<(kPeM <= 16), tileAM16, tileAM32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    constexpr int kCellRegBytes = 256 * 1024;
    constexpr int kAccumulatorTileRegCount = 4;
    constexpr int kTileABytes = tileA::LogicalTileBytes;
    constexpr int kTileCBytes = tileC::LogicalTileBytes;
    constexpr int kCByBytes = kTileABytes < kCellRegBytes
        ? (kCellRegBytes - kTileABytes) / kTileCBytes : 0;
    constexpr int kCGroupCapacity = kCByBytes < kAccumulatorTileRegCount
        ? kCByBytes : kAccumulatorTileRegCount;
    constexpr int kReuseN = Nb < kCGroupCapacity ? Nb : kCGroupCapacity;

    static_assert(ValidM >= 1 && ValidM <= 128);
    static_assert(kReuseN > 0,
                  "CellReg must hold one local A tile and one C tile");
    static_assert(tileBMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tile exceeds the 256 KiB SharedTReg pool");
    static_assert(kTileABytes + kReuseN * kTileCBytes <= kCellRegBytes,
                  "A and resident C group exceed CellReg capacity");

    const int tid = static_cast<int>(get_thread_idx());
#pragma clang loop unroll(full)
    for (int jBase = 0; jBase < Nb; jBase += kReuseN) {
        MatmulReuseAStorage<tileC, kReuseN> tCGroup;
        // Separate first/middle/final K so the two N accumulator chains do
        // not merge through loop-internal first/last-K control-flow PHIs.
        run_full_n_k<true, Kb == 1, kReuseN, dtype,
                     gM, gN, gK, tN, tK, ValidM>(
            a_ptr, b_ptr, tCGroup, m_begin, jBase, 0, tid);
#pragma clang loop unroll(disable)
        for (int k = 1; k < Kb - 1; ++k) {
            run_full_n_k<false, false, kReuseN, dtype,
                         gM, gN, gK, tN, tK, ValidM>(
                a_ptr, b_ptr, tCGroup, m_begin, jBase, k, tid);
        }
        if constexpr (Kb > 1) {
            run_full_n_k<false, true, kReuseN, dtype,
                         gM, gN, gK, tN, tK, ValidM>(
                a_ptr, b_ptr, tCGroup, m_begin, jBase, Kb - 1, tid);
        }
#pragma clang loop unroll(full)
        for (int jj = 0; jj < kReuseN; ++jj) {
            if (jBase + jj < Nb) {
                gmC gC(c_ptr + (m_begin + local_m_offset<ValidM, kPeM>(tid)) * gN +
                       (jBase + jj) * tN);
                store_c<ValidM>(gC, tCGroup[jj], tid);
            }
        }
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int ValidN>
__attribute__((noinline)) void run_tail_block(float *c_ptr, dtype *a_ptr,
                                      dtype *b_ptr, int m_begin,
                                      int n_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int Kb = gK / tK;
    constexpr int ComputeM = compute_group_m(ValidM);
    constexpr auto matmulOptions = fixp::keep_acc();
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using tileAM16 = CubeTileM16<dtype, kPeM, tK>;
    using tileAM32 = CubeTileM32<dtype, kPeM, tK>;
    using tileA = std::conditional_t<(kPeM <= 16), tileAM16, tileAM32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
    using tileB = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN, kPeM, ValidN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN, kPeM, ValidN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    static_assert(ValidM >= 1 && ValidM <= 128);
    static_assert(ValidN >= 1 && ValidN <= tN);
    static_assert(tileBMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tail tile exceeds the 256 KiB SharedTReg pool");
    static_assert(tileA::LogicalTileBytes + tileC::LogicalTileBytes <=
                      256 * 1024,
                  "A and tail C tile exceed CellReg capacity");

    const int tid = static_cast<int>(get_thread_idx());
    tileC tC;
    // Tail blocks follow the same cross-K local-A buffer reuse as full N.
    tileA tA;
#pragma clang loop unroll(disable)
    for (int k = 0; k < Kb; ++k) {
        // Every PE must publish a Local-A primary before the cooperative
        // TMATMUL. Inactive tail PEs load a safe dummy shard; groupM masks
        // their rows and no C store is issued for them.
        const int local_m = m_begin + local_m_offset<ValidM, kPeM>(tid);
        gmA gA(a_ptr + local_m * gK + k * tK);
        TLOAD_CUBE(tA, gA);
        tileB tB;
        gmB gB(b_ptr + n_begin * gK + k * tK);
        TLOAD<tileBMatrix, 1>(tB, gB);
        if constexpr (Kb == 1) {
            TMATMUL(tC, tA, tB, matmulOptions, ComputeM);
        } else if (k == 0) {
            TMATMUL(tC, tA, tB, matmulOptions.raw_acc(), ComputeM);
        } else if (k == Kb - 1) {
            TMATMUL_ACC(tC, tC, tA, tB, matmulOptions.acc_hint(),
                        ComputeM);
        } else {
            TMATMUL_ACC(tC, tC, tA, tB,
                        matmulOptions.raw_acc().acc_hint(), ComputeM);
        }
    }
    gmC gC(c_ptr + (m_begin + local_m_offset<ValidM, kPeM>(tid)) * gN + n_begin);
    store_c<ValidM>(gC, tC, tid);
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int RemainingN>
PTO_SHARED_INLINE void run_n_tail(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin, int n_begin) {
    constexpr int kChunkN = floor_power_of_two(RemainingN);
    run_tail_block<dtype, gM, gN, gK, tN, tK, ValidM, kChunkN>(
        c_ptr, a_ptr, b_ptr, m_begin, n_begin);
    if constexpr (RemainingN > kChunkN) {
        run_n_tail<dtype, gM, gN, gK, tN, tK, ValidM,
                   RemainingN - kChunkN>(
            c_ptr, a_ptr, b_ptr, m_begin, n_begin + kChunkN);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_m_strip(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                   int m_begin) {
    constexpr int kFullNBlocks = gN / tN;
    constexpr int kTailN = gN % tN;
    if constexpr (kFullNBlocks > 0) {
        run_full_n<dtype, gM, gN, gK, tN, tK, ValidM>(
            c_ptr, a_ptr, b_ptr, m_begin);
    }
    if constexpr (kTailN != 0) {
        run_n_tail<dtype, gM, gN, gK, tN, tK, ValidM, kTailN>(
            c_ptr, a_ptr, b_ptr, m_begin, (gN / tN) * tN);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int RemainingM>
__attribute__((noinline)) void run_m_tail(float *c_ptr, dtype *a_ptr,
                                          dtype *b_ptr, int m_begin) {
    constexpr int ChunkM = RemainingM > 64 ? RemainingM : floor_power_of_two(RemainingM);
    run_m_strip<dtype, gM, gN, gK, tN, tK, ChunkM>(
        c_ptr, a_ptr, b_ptr, m_begin);
    if constexpr (RemainingM > ChunkM) {
        run_m_tail<dtype, gM, gN, gK, tN, tK, RemainingM - ChunkM>(
            c_ptr, a_ptr, b_ptr, m_begin + ChunkM);
    }
}

}  // namespace matmul_reuse_a_detail

// Four-PE cooperative matmul with PE-local A tile storage reused across K:
// each K iteration refills the same logical CellReg tile with a different A
// slice. Complete N blocks can also reuse that loaded A slice across up to
// four resident C accumulators, including M16 tails. M tails >64 stay one
// M128 compute group using overlapping A shards and disjoint masked stores.
// Smaller M tails and N tails retain power-of-two chunks.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseA(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert((kGroupM & (kGroupM - 1)) == 0,
                  "group_M must be a power of two");
    static_assert((tN & (tN - 1)) == 0,
                  "tN must be a power of two");

    constexpr int kFullMBlocks = gM / kGroupM;
    constexpr int kTailM = gM % kGroupM;
#pragma clang loop unroll(full)
    for (int i = 0; i < kFullMBlocks; ++i) {
        matmul_reuse_a_detail::run_m_strip<
            dtype, gM, gN, gK, tN, tK, kGroupM>(
            c_ptr, a_ptr, b_ptr, i * kGroupM);
    }
    if constexpr (kTailM != 0) {
        matmul_reuse_a_detail::run_m_tail<
            dtype, gM, gN, gK, tN, tK, kTailM>(
            c_ptr, a_ptr, b_ptr, kFullMBlocks * kGroupM);
    }
}
