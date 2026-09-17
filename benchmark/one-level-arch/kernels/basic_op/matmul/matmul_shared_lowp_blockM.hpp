#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Four-PE low-precision matmul that batches multiple M-tile and/or N-tile
// loads and shares the opposite operand across them.
//
// Loop order and sharing:
//   for each N-block-group (step blockN)
//     for each M-block-group (step blockM)
//       for each K block
//         load blockM A tiles + blockM A scale tiles (shared across N)
//         load blockN B tiles + blockN B scale tiles (shared across M)
//         accumulate C[mi][nj] for mi,nj in block
//       store blockM * blockN C tiles
//
// B is shared across M-tiles (blockM > 1), A is shared across N-tiles
// (blockN > 1).  Either or both can be 1.
//
// A/B are Core-shared operands. Each PE owns one [tM / 4, tN] FP32 C tile.
// PackedFactor is 1 for 8-bit formats and 2 for packed FP4x2 formats.
// MXFP8/MXFP4 additionally load group-32 E8M0 scale data through Shared
// tiles. HiF4X2 is a Matrix-MX-only input and uses one raw U32 scale word
// per 64 logical K elements. HiF8 is a normal TMATMUL input in PTO v0.58
// and does not take MX scales.
template <typename dtype, typename scale_dtype, int PackedFactor, bool UseMx,
          int ScaleGroup,
          int gM, int gN, int gK, int tM, int tN, int tK,
          int blockM, int blockN>
void matmul_shared_lowp_blockM(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                scale_dtype *a_scale_ptr,
                                scale_dtype *b_scale_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kValidRowM = (gM < kTileRows) ? gM : kTileRows;
    constexpr int kPeM = kValidRowM <= 64 ? 16 : 32;
    constexpr int kStoredGK = gK / PackedFactor;
    constexpr int kStoredTK = tK / PackedFactor;
    constexpr int kScaleK = (tK + ScaleGroup - 1) / ScaleGroup;
    constexpr int kPaddedScaleK = ((kScaleK + 31) / 32) * 32;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8) or 2 (FP4x2)");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(gK % PackedFactor == 0 && tK % PackedFactor == 0,
                  "K must be divisible by the packed element factor");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert(gM % kGroupM == 0 || gM < kGroupM,
                  "gM must be a multiple of kGroupM or smaller than it");
    static_assert(gM % kPeM == 0 || gM < kGroupM,
                  "gM must be a multiple of kPeM, or a partial tile (gM < kGroupM)");
    static_assert(gM % (blockM * kGroupM) == 0 || (gM < kGroupM && blockM == 1),
                  "gM must be divisible by blockM * kGroupM, or partial tile with blockM=1");
    static_assert(gN % (blockN * tN) == 0,
                  "gN must be divisible by blockN * tN");
    static_assert(blockM >= 1, "blockM must be at least 1");
    static_assert(gM >= kGroupM || blockM == 1,
                  "blockM > 1 requires gM >= kGroupM");
    static_assert(blockN >= 1, "blockN must be at least 1");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");
    static_assert(!UseMx || (gK % ScaleGroup == 0 &&
                             tK % ScaleGroup == 0),
                  "MX formats require K and tK divisible by their scale group");

    // K chain: single=0, begin=raw_acc, middle=raw_acc|acc_hint, end=acc_hint.
    // C remains explicit; gfsim also needs cube.enable_internal_acc=true.
    constexpr auto matmulOptions = fixp::keep_acc().transpose_b();
    const uint32_t tid = get_thread_idx();

    using gmASlice = global_tensor<dtype, RowMajor<kGroupM, kStoredGK>>;
    using gmBSlice = global_tensor<dtype, RowMajor<kStoredTK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    using tileAMatrix = SharedMatrixLeft<dtype, kTileRows, tK, kValidRowM, tK>;
    using tileBMatrix = SharedMatrixRight<dtype, tK, tN>;
    using tileAShared = SharedTile<tileAMatrix>;
    using tileBShared = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    using itC = global_iterator<gmC, tileC>;

    itC gIterC(c_ptr);

    using gmAScale =
        global_tensor<scale_dtype, RowMajor<gM, gK / ScaleGroup>>;
    using gmBScale =
        global_tensor<scale_dtype, RowMajor<gK / ScaleGroup, gN>>;
    using tileAScaleMatrix =
        SharedMatrixLeft<scale_dtype, kTileRows, kPaddedScaleK,
                         kValidRowM, kScaleK>;
    using tileBScaleMatrix =
        SharedMatrixRight<scale_dtype, kPaddedScaleK, tN,
                          kScaleK, tN>;
    using tileAScale = SharedTile<tileAScaleMatrix>;
    using tileBScale = SharedTile<tileBScaleMatrix>;
    using itAScale = global_iterator<gmAScale, tileAScaleMatrix>;
    using itBScale = global_iterator<gmBScale, tileBScaleMatrix>;

    // SharedTReg provides 256 KiB. The block holds blockM A tiles (plus
    // blockM A scale tiles for MX) and blockN B tiles (plus blockN B scale
    // tiles for MX).
    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kTileABytes = tileAMatrix::LogicalTileBytes;
    constexpr int kTileBBytes = tileBMatrix::LogicalTileBytes;
    constexpr int kTileAScaleBytes = UseMx ? tileAScaleMatrix::LogicalTileBytes : 0;
    constexpr int kTileBScaleBytes = UseMx ? tileBScaleMatrix::LogicalTileBytes : 0;
    constexpr int kBlockSharedBytes =
        blockM * (kTileABytes + kTileAScaleBytes) +
        blockN * (kTileBBytes + kTileBScaleBytes);
    static_assert(kBlockSharedBytes <= kSharedTRegBytes,
                  "block tiles exceed the 256 KiB SharedTReg pool");

    itAScale gIterAScale(a_scale_ptr);
    itBScale gIterBScale(b_scale_ptr);

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
                    gmASlice gA(a_ptr + (bi * blockM + mi) * kGroupM *
                                        kStoredGK + k * kStoredTK);
                    TLOAD<tileAMatrix, 1>(tA[mi], gA);
                }

                // Load blockN B tiles — each shared across blockM M-tiles.
                tileBShared tB[blockN];
#pragma clang loop unroll(full)
                for (int nj = 0; nj < blockN; ++nj) {
                    gmBSlice gB(b_ptr + k * kStoredTK * gN +
                                         (bj * blockN + nj) * tN);
                    TLOAD<tileBMatrix, 1>(tB[nj], gB);
                }

                if constexpr (UseMx) {
                    // Load blockM A scale tiles + blockN B scale tiles.
                    tileAScale tAScale[blockM];
                    tileBScale tBScale[blockN];
#pragma clang loop unroll(full)
                    for (int mi = 0; mi < blockM; ++mi) {
                        auto gAScale = gIterAScale(bi * blockM + mi, k);
                        TLOAD<tileAScaleMatrix, 1>(tAScale[mi], gAScale);
                    }
#pragma clang loop unroll(full)
                    for (int nj = 0; nj < blockN; ++nj) {
                        auto gBScale = gIterBScale(k, bj * blockN + nj);
                        TLOAD<tileBScaleMatrix, 1>(tBScale[nj], gBScale);
                    }

#pragma clang loop unroll(full)
                    for (int nj = 0; nj < blockN; ++nj) {
#pragma clang loop unroll(full)
                        for (int mi = 0; mi < blockM; ++mi) {
                            if constexpr (Kb == 1) {
                                TMATMUL_MX<3>(tC[mi * blockN + nj], tA[mi], tAScale[mi], tB[nj], tBScale[nj], matmulOptions);
                            } else if (k == 0) {
                                TMATMUL_MX<3>(tC[mi * blockN + nj], tA[mi], tAScale[mi], tB[nj], tBScale[nj], matmulOptions.raw_acc());
                            } else if (k == Kb - 1) {
                                TMATMUL_MX_ACC<3>(tC[mi * blockN + nj], tC[mi * blockN + nj], tA[mi], tAScale[mi], tB[nj], tBScale[nj], matmulOptions.acc_hint());
                            } else {
                                TMATMUL_MX_ACC<3>(tC[mi * blockN + nj], tC[mi * blockN + nj], tA[mi], tAScale[mi], tB[nj], tBScale[nj], matmulOptions.raw_acc().acc_hint());
                            }
                        }
                    }
                } else {
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
            }

            // Store blockM * blockN C tiles.
#pragma clang loop unroll(full)
            for (int mi = 0; mi < blockM; ++mi) {
#pragma clang loop unroll(full)
                for (int nj = 0; nj < blockN; ++nj) {
                    if constexpr (gM >= kGroupM) {
                        auto gC = gIterC((bi * blockM + mi) * kPeNum + tid,
                                         bj * blockN + nj);
                        TSTORE_CUBE(gC, tC[mi * blockN + nj]);
                    } else {
                        if (((bi * blockM + mi) * kPeNum + static_cast<int>(tid)) * kPeM < gM) {
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
