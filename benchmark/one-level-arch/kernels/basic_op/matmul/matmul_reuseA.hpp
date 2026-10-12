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

// Four-PE cooperative matmul with PE-local A reuse.
//
// Each PE loads only its 16/32-row shard of A into a local CUBE tile
// (CellReg). That A tile is reused across a capacity-sized group of N blocks,
// whose C accumulators remain in CellReg across the complete K chain. B stays
// a cooperative Shared-Right operand. The effective loop/load order is:
//
//   for each M group
//     for each capacity-sized N group
//       keep the group's C accumulators in CellReg
//       for each K block
//         load one local A tile
//         for each N block in the group
//           load shared B
//           accumulate C = A * B^T
//
// Thus one A load serves up to four N blocks. The external operand contract is
// unchanged: A is [M,K], B is B-major [N,K], and C = A * B^T.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseA(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kValidRowM = (gM < kTileRows) ? gM : kTileRows;
    constexpr int kPeM = kValidRowM <= 64 ? 16 : 32;

    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert(gM % kGroupM == 0 || gM < kGroupM,
                  "gM must be a multiple of group_M or smaller than it");
    static_assert(gM % kPeM == 0 || gM < kGroupM,
                  "gM must be a multiple of the per-PE M shard");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    const uint32_t tid = get_thread_idx();

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

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = (gM + kGroupM - 1) / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    // CellReg is private to each PE: 2048 x 128 B = 256 KiB. One local A tile
    // and a group of C accumulators are live together. The accumulator bank
    // has four encodable names (M#/N#/T#/U#), so the reuse group is bounded by
    // both CellReg bytes and architectural names.
    constexpr int kCellRegBytes = 256 * 1024;
    constexpr int kAccumulatorTileRegCount = 4;
    constexpr int kTileABytes = tileA::LogicalTileBytes;
    constexpr int kTileCBytes = tileC::LogicalTileBytes;
    constexpr int kCByBytes =
        kTileABytes < kCellRegBytes
            ? (kCellRegBytes - kTileABytes) / kTileCBytes
            : 0;
    constexpr int kCGroupCapacity =
        kCByBytes < kAccumulatorTileRegCount ? kCByBytes
                                             : kAccumulatorTileRegCount;
    constexpr int kReuseN = Nb < kCGroupCapacity ? Nb : kCGroupCapacity;

    static_assert(kReuseN > 0,
                  "CellReg must hold one local A tile and one C tile");
    static_assert(tileBMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tile exceeds the 256 KiB SharedTReg pool");
    static_assert(kTileABytes + kReuseN * kTileCBytes <= kCellRegBytes,
                  "A and resident C group exceed CellReg capacity");

#pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
        for (int jBase = 0; jBase < Nb; jBase += kReuseN) {
            MatmulReuseAStorage<tileC, kReuseN> tCGroup;

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                tileA tA;
                // gIterA advances by kPeM rows; four consecutive iterator
                // rows form one cooperative group_M block.
                if constexpr (gM >= kGroupM) {
                    auto gA = gIterA(i * kPeNum + tid, k);
                    TLOAD_CUBE(tA, gA);
                } else if (static_cast<int>(tid) * kPeM < gM) {
                    auto gA = gIterA(tid, k);
                    TLOAD_CUBE(tA, gA);
                }

#pragma clang loop unroll(full)
                for (int jj = 0; jj < kReuseN; ++jj) {
                    if (jBase + jj < Nb) {
                        tileB tB;
                        auto gB = gIterB(jBase + jj, k);
                        TLOAD<tileBMatrix, 1>(tB, gB);

                        if constexpr (Kb == 1) {
                            TMATMUL(tCGroup[jj], tA, tB, kValidRowM);
                        } else if (k == 0) {
                            TMATMUL(tCGroup[jj], tA, tB, kValidRowM);
                        } else {
                            TMATMUL_ACC(tCGroup[jj], tCGroup[jj], tA, tB,
                                        kValidRowM);
                        }
                    }
                }
            }

#pragma clang loop unroll(full)
            for (int jj = 0; jj < kReuseN; ++jj) {
                if (jBase + jj < Nb) {
                    if constexpr (gM >= kGroupM) {
                        auto gC = gIterC(i * kPeNum + tid, jBase + jj);
                        TSTORE_CUBE(gC, tCGroup[jj]);
                    } else if (static_cast<int>(tid) * kPeM < gM) {
                        auto gC = gIterC(tid, jBase + jj);
                        TSTORE_CUBE(gC, tCGroup[jj]);
                    }
                }
            }
        }
    }
}
