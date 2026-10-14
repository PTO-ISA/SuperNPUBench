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

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_full_n(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr auto matmulOptions = fixp::keep_acc();
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using tileAM16 = CubeTileM16<dtype, kPeM, tK>;
    using tileAM32 = CubeTileM32<dtype, kPeM, tK>;
    using tileA = std::conditional_t<(kPeM <= 16), tileAM16, tileAM32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileB = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;
    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

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
    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);
#pragma clang loop unroll(full)
    for (int jBase = 0; jBase < Nb; jBase += kReuseN) {
        MatmulReuseAStorage<tileC, kReuseN> tCGroup;
        // Refill the same PE-local A tile for each K slice. Keeping the K loop
        // rolled prevents a distinct logical A tile from being instantiated
        // for every unrolled K iteration.
        tileA tA;
#pragma clang loop unroll(disable)
        for (int k = 0; k < Kb; ++k) {
            if constexpr (ValidM >= kPeM * 4) {
                auto gA = gIterA(m_begin / kPeM + tid, k);
                TLOAD_CUBE(tA, gA);
            } else if (tid * kPeM < ValidM) {
                auto gA = gIterA(m_begin / kPeM + tid, k);
                TLOAD_CUBE(tA, gA);
            }
#pragma clang loop unroll(full)
            for (int jj = 0; jj < kReuseN; ++jj) {
                if (jBase + jj < Nb) {
                    tileB tB;
                    auto gB = gIterB(jBase + jj, k);
                    TLOAD<tileBMatrix, 1>(tB, gB);
                    if constexpr (Kb == 1) {
                        TMATMUL(tCGroup[jj], tA, tB, matmulOptions,
                                ValidM);
                    } else if (k == 0) {
                        TMATMUL(tCGroup[jj], tA, tB,
                                matmulOptions.raw_acc(), ValidM);
                    } else if (k == Kb - 1) {
                        TMATMUL_ACC(tCGroup[jj], tCGroup[jj], tA, tB,
                                    matmulOptions.acc_hint(), ValidM);
                    } else {
                        TMATMUL_ACC(tCGroup[jj], tCGroup[jj], tA, tB,
                                    matmulOptions.raw_acc().acc_hint(),
                                    ValidM);
                    }
                }
            }
        }
#pragma clang loop unroll(full)
        for (int jj = 0; jj < kReuseN; ++jj) {
            if (jBase + jj < Nb) {
                if constexpr (ValidM >= kPeM * 4) {
                    auto gC = gIterC(m_begin / kPeM + tid, jBase + jj);
                    TSTORE_CUBE(gC, tCGroup[jj]);
                } else if (tid * kPeM < ValidM) {
                    auto gC = gIterC(m_begin / kPeM + tid, jBase + jj);
                    TSTORE_CUBE(gC, tCGroup[jj]);
                }
            }
        }
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int ValidN>
PTO_SHARED_INLINE void run_tail_block(float *c_ptr, dtype *a_ptr,
                                      dtype *b_ptr, int m_begin,
                                      int n_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int Kb = gK / tK;
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
        const int local_m = tid * kPeM < ValidM
            ? m_begin + tid * kPeM : m_begin;
        gmA gA(a_ptr + local_m * gK + k * tK);
        TLOAD_CUBE(tA, gA);
        tileB tB;
        gmB gB(b_ptr + n_begin * gK + k * tK);
        TLOAD<tileBMatrix, 1>(tB, gB);
        if constexpr (Kb == 1) {
            TMATMUL(tC, tA, tB, matmulOptions, ValidM);
        } else if (k == 0) {
            TMATMUL(tC, tA, tB, matmulOptions.raw_acc(), ValidM);
        } else if (k == Kb - 1) {
            TMATMUL_ACC(tC, tC, tA, tB, matmulOptions.acc_hint(),
                        ValidM);
        } else {
            TMATMUL_ACC(tC, tC, tA, tB,
                        matmulOptions.raw_acc().acc_hint(), ValidM);
        }
    }
    if (tid * kPeM < ValidM) {
        gmC gC(c_ptr + (m_begin + tid * kPeM) * gN + n_begin);
        TSTORE_CUBE(gC, tC);
    }
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
    if constexpr (ValidM > 64) {
        run_full_n<dtype, gM, gN, gK, tN, tK, ValidM>(
            c_ptr, a_ptr, b_ptr, m_begin);
    } else {
        // A short M tail uses M16 accumulators. Process one N block at a time
        // so the compiler does not have to move an M16 accumulator array
        // through register names already used by the M32 full-M path.
#pragma clang loop unroll(disable)
        for (int j = 0; j < kFullNBlocks; ++j) {
            run_tail_block<dtype, gM, gN, gK, tN, tK, ValidM, tN>(
                c_ptr, a_ptr, b_ptr, m_begin, j * tN);
        }
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
    constexpr int kChunkM = floor_power_of_two(RemainingM);
    run_m_strip<dtype, gM, gN, gK, tN, tK, kChunkM>(
        c_ptr, a_ptr, b_ptr, m_begin);
    if constexpr (RemainingM > kChunkM) {
        run_m_tail<dtype, gM, gN, gK, tN, tK, RemainingM - kChunkM>(
            c_ptr, a_ptr, b_ptr, m_begin + kChunkM);
    }
}

}  // namespace matmul_reuse_a_detail

// Four-PE cooperative matmul with PE-local A tile storage reused across K:
// each K iteration refills the same logical CellReg tile with a different A
// slice. Complete N blocks can also reuse that loaded A slice across up to
// four resident C accumulators. Non-divisible M and N extents are split into
// power-of-two tail chunks whose tile types carry valid_row/valid_col.
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
