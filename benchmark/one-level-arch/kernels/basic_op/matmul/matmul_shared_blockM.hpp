#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE cooperative matmul that batches multiple M-tile and/or N-tile loads
// and shares the opposite operand across them.
//
// Loop order and sharing:
//   for each N-block-group (step blockN)
//     for each M-block-group (step blockM)
//       for each K block
//         load blockM A tiles (shared across the blockN N-tiles)
//         load blockN B tiles (shared across the blockM M-tiles)
//         accumulate C[mi][nj] = A[mi] * B[nj]  for mi,nj in block
//       store blockM * blockN C tiles
//
// B is shared across M-tiles (blockM > 1), A is shared across N-tiles
// (blockN > 1).  Either or both can be 1 (no sharing on that axis).
// blockM=1,blockN=1 degenerates to the original matmul_shared.
//
// A and B are cooperative SharedTile operands. Each PE owns one contiguous
// CUBE row slice of the [group_M, tN] FP32 output tile.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK,
          int blockM, int blockN>
void matmul_shared_blockM(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
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
    static_assert(gM >= kGroupM || blockM == 1,
                  "blockM > 1 requires gM >= kGroupM");
    static_assert(gM % (blockM * kGroupM) == 0 ||
                  (gM < kGroupM && blockM == 1),
                  "gM must be divisible by blockM * group_M "
                  "(or be a single partial tile with blockM=1)");
    static_assert(gN % (blockN * tN) == 0,
                  "gN must be divisible by blockN * tN");
    static_assert(blockM >= 1, "blockM must be at least 1");
    static_assert(blockN >= 1, "blockN must be at least 1");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    // K chain: single=0, begin=raw_acc, middle=raw_acc|acc_hint, end=acc_hint.
    // C remains explicit; gfsim also needs cube.enable_internal_acc=true.
    constexpr auto matmulOptions = fixp::keep_acc();
    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    // B is stored pre-transposed as the physical [N, K] rectangle expected
    // by a non-transposed cooperative Shared right primary.
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kTileRows, tK, kValidRowM, tK>;
    // The current cooperative TMATMUL contract declares a non-transposed
    // Shared B primary in its physical RowMajor [N, K] shape.
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    // SharedTReg provides 256 KiB for the cooperative SharedTile operands.
    // The block holds blockM A tiles plus blockN B tiles.
    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kTileABytes = tileAMatrix::LogicalTileBytes;
    constexpr int kTileBBytes = tileBMatrix::LogicalTileBytes;
    constexpr int kBlockSharedBytes = blockM * kTileABytes + blockN * kTileBBytes;
    static_assert(kBlockSharedBytes <= kSharedTRegBytes,
                  "blockM A tiles plus blockN B tiles exceed the 256 KiB SharedTReg pool");

    using itA = global_iterator<gmA, tileAMatrix>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = (gM + kGroupM - 1) / kGroupM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    constexpr int Mbi = Mb / blockM;
    constexpr int Nbj = Nb / blockN;

#pragma clang loop unroll(full)
    for (int bj = 0; bj < Nbj; ++bj) {
#pragma clang loop unroll(full)
        for (int bi = 0; bi < Mbi; ++bi) {
            tileC tC[blockM * blockN];

#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                // Load blockM A tiles — each shared across blockN N-tiles.
                tileAShared tA[blockM];
#pragma clang loop unroll(full)
                for (int mi = 0; mi < blockM; ++mi) {
                    auto gA = gIterA(bi * blockM + mi, k);
                    TLOAD<tileAMatrix, 1>(tA[mi], gA);
                }

                // Load blockN B tiles — each shared across blockM M-tiles.
                tileBShared tB[blockN];
#pragma clang loop unroll(full)
                for (int nj = 0; nj < blockN; ++nj) {
                    auto gB = gIterB(bj * blockN + nj, k);
                    TLOAD<tileBMatrix, 1>(tB[nj], gB);
                }

                // Compute blockM * blockN cooperative TMATMULs.
                // Inner loop over N so the same B tile is reused by
                // consecutive TMATMULs (mirrors the blockM-only pattern).
#pragma clang loop unroll(full)
                for (int nj = 0; nj < blockN; ++nj) {
#pragma clang loop unroll(full)
                    for (int mi = 0; mi < blockM; ++mi) {
                        if constexpr (Kb == 1) {
                            TMATMUL(tC[mi * blockN + nj], tA[mi], tB[nj], matmulOptions);
                        } else if (k == 0) {
                            TMATMUL(tC[mi * blockN + nj], tA[mi], tB[nj], matmulOptions.raw_acc());
                        } else if (k == Kb - 1) {
                            TMATMUL_ACC(tC[mi * blockN + nj], tC[mi * blockN + nj], tA[mi], tB[nj], matmulOptions.acc_hint());
                        } else {
                            TMATMUL_ACC(tC[mi * blockN + nj], tC[mi * blockN + nj], tA[mi], tB[nj], matmulOptions.raw_acc().acc_hint());
                        }
                    }
                }
            }

            // Store blockM * blockN C tiles.  When gM < kGroupM only some
            // PEs have valid rows in the single partial M-tile.
#pragma clang loop unroll(full)
            for (int mi = 0; mi < blockM; ++mi) {
#pragma clang loop unroll(full)
                for (int nj = 0; nj < blockN; ++nj) {
                    if constexpr (gM >= kGroupM) {
                        auto gC = gIterC((bi * blockM + mi) * kPeNum + tid,
                                         bj * blockN + nj);
                        TSTORE_CUBE(gC, tC[mi * blockN + nj]);
                    } else {
                        if (((bi * blockM + mi) * kPeNum +
                             static_cast<int>(tid)) * kPeM < gM) {
                            auto gC = gIterC((bi * blockM + mi) * kPeNum + tid,
                                             bj * blockN + nj);
                            TSTORE_CUBE(gC, tC[mi * blockN + nj]);
                        }
                    }
                }
            }
        }
    }
}
