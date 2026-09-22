#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

template <typename TileT, int Count>
struct MatmulReuseBStorage {
    TileT tiles[Count];

    TileT &operator[](int index) { return tiles[index]; }
};

template <typename TileT>
struct MatmulReuseBStorage<TileT, 0> {};

// Four-PE cooperative matmul that keeps Shared B tiles alive across M blocks.
//
// Loop order and reuse:
//   for each N block
//     for each M block
//       for each K block
//         load A for the current M block
//         load B only for the first M block, then reuse it
//         accumulate C = A * B^T
//
// A and B are cooperative SharedTile operands. Each PE owns one contiguous
// CUBE row slice of the [group_M, tN] FP32 output tile.
// The external operand contract matches matmul_shared: A is [M,K], B is
// B-major [N,K], and C = A * B^T.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_shared_reuseB(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
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
                  "gM must be a multiple of kGroupM or smaller than it");
    static_assert(gM % kPeM == 0 || gM < kGroupM,
                  "gM must be a multiple of kPeM, or a partial tile (gM < kGroupM)");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    // K chain: single=0, begin=raw_acc, middle=raw_acc|acc_hint, end=acc_hint.
    // C remains explicit; gfsim also needs cube.enable_internal_acc=true.
    constexpr auto matmulOptions = fixp::keep_acc();
    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kTileRows, tK, kValidRowM, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    // SharedTReg provides 256 KiB for the cooperative SharedTile operands.
    // Reserve the A and B operands needed by the current multiply first. Only
    // the SharedTReg capacity left after those two active tiles may be used by
    // B tiles that stay resident across M blocks. LogicalTileBytes includes
    // TileOP storage-capacity rounding, rather than only valid elements.
    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kTileABytes = tileAMatrix::LogicalTileBytes;
    constexpr int kTileBBytes = tileBMatrix::LogicalTileBytes;
    constexpr int kActiveOperandBytes = kTileABytes + kTileBBytes;
    static_assert(kActiveOperandBytes <= kSharedTRegBytes,
                  "SharedTReg must hold the active tileA and tileB");
    constexpr int kMaxReuseBTiles =
        (kSharedTRegBytes - kActiveOperandBytes) / kTileBBytes;

    using itA = global_iterator<gmA, tileAMatrix>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = (gM + kGroupM - 1) / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr int kReuseK =
        Kb < kMaxReuseBTiles ? Kb : kMaxReuseBTiles;

#pragma clang loop unroll(full)
    for (int j = 0; j < Nb; ++j) {
        // These Shared B handles span the complete M loop. The first M block
        // fills them while later M blocks consume the same Shared versions.
        MatmulReuseBStorage<tileBShared, kReuseK> tBReuse;

#pragma clang loop unroll(full)
        for (int i = 0; i < Mb; ++i) {
            tileC tC;

            if constexpr (kReuseK > 0) {
#pragma clang loop unroll(full)
                for (int k = 0; k < kReuseK; ++k) {
                    tileAShared tAShared;
                    auto gA = gIterA(i, k);
                    TLOAD<tileAMatrix, 1>(tAShared, gA);

                    if (i == 0) {
                        auto gB = gIterB(j, k);
                        TLOAD<tileBMatrix, 1>(tBReuse[k], gB);
                    }

                    if constexpr (Kb == 1) {
                        TMATMUL(tC, tAShared, tBReuse[k], matmulOptions);
                    } else if (k == 0) {
                        TMATMUL(tC, tAShared, tBReuse[k], matmulOptions.raw_acc());
                    } else if (k == Kb - 1) {
                        TMATMUL_ACC(tC, tC, tAShared, tBReuse[k], matmulOptions.acc_hint());
                    } else {
                        TMATMUL_ACC(tC, tC, tAShared, tBReuse[k], matmulOptions.raw_acc().acc_hint());
                    }
                }
            }

            // K blocks beyond the reuse window preserve the original compute
            // path and reload both operands for every M block.
            if constexpr (kReuseK < Kb) {
#pragma clang loop unroll(full)
                for (int k = kReuseK; k < Kb; ++k) {
                    tileAShared tAShared;
                    tileBShared tBShared;
                    auto gA = gIterA(i, k);
                    auto gB = gIterB(j, k);
                    TLOAD<tileAMatrix, 1>(tAShared, gA);
                    TLOAD<tileBMatrix, 1>(tBShared, gB);
                    if constexpr (Kb == 1) {
                        TMATMUL(tC, tAShared, tBShared, matmulOptions);
                    } else if (k == 0) {
                        TMATMUL(tC, tAShared, tBShared, matmulOptions.raw_acc());
                    } else if (k == Kb - 1) {
                        TMATMUL_ACC(tC, tC, tAShared, tBShared, matmulOptions.acc_hint());
                    } else {
                        TMATMUL_ACC(tC, tC, tAShared, tBShared, matmulOptions.raw_acc().acc_hint());
                    }
                }
            }

            if constexpr (gM >= kGroupM) {
                auto gC = gIterC(i * kPeNum + tid, j);
                TSTORE_CUBE(gC, tC);
            } else {
                if ((i * kPeNum + static_cast<int>(tid)) * kPeM < gM) {
                    auto gC = gIterC(i * kPeNum + tid, j);
                    TSTORE_CUBE(gC, tC);
                }
            }
        }
    }
}
