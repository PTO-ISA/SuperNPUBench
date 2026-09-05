#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// 4-PE cooperative TMATMUL FlashAttention.
//
// O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// Q/K/V are staged into shared tile storage via TLOAD (PEMask=1), matching
// multi_thread/matmul/matmul_shared.hpp. get_thread_idx() selects the current
// PE's output row range; the four-PE group row split is implicit.
//
// Two layout optimizations keep P in tile registers across P*V, avoiding the
// GM round-trip used by the earlier shared-shared form:
//
// 1. Score->P dtype cast: TCVT directly from the Acc CubeM32 score tile
//    (float) to a CubeTileM32 (Location::Left) matrix_dtype tile -- a
//    cross-Location, same-layout, pure-dtype cast. No Vec RowMajor bridge.
//
// 2. P*V direct local-Left: the P shard is a local CUBE Left operand in
//    CubeM32 (matching the TMATMUL left-operand contract), so it feeds P*V
//    directly without TSTORE_CUBE, GM scratch, or SharedTile reload. Each PE
//    keeps its own [kPeTm, kStoredTk] P shard in tile registers and multiplies
//    it against the shared V Right tile (Local-A + Shared-B, allowed by
//    validate_matrix_contract). The entire TSTORE_CUBE -> GM -> TLOAD ->
//    SharedTile path is eliminated.
//
// (MX mode retains the GM round-trip fallback for P scale loading.)

// Convert two logical scalar columns into one packed-x2 cube element.
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
inline void fa_tcvt_packed_x2(tile_shape_out &dst, tile_shape_in &src) {
    static_assert(tile_shape_out::Rows == tile_shape_in::Rows,
                  "packed TCVT must preserve rows");
    static_assert(tile_shape_in::Cols == tile_shape_out::Cols * 2,
                  "packed-x2 TCVT destination must have half as many columns");
    const size_t valid_col = dst.GetValidCol();
    const size_t valid_row = dst.GetValidRow();
    asm volatile(
        "BSTART.TEPL 27, %c1\n"
        "B.DATR %c2, RNone\n"
        "B.IOT %3, mask=15, last, ->%0<%Z4>\n"
        "B.DIM %5, 0, ->lb0\n"
        "B.DIM %6, 0, ->lb1\n"
        "B.DIM zero, %c7, ->lb2\n"
        : "=Tr"(dst.data())
        : "i"(type_traits<typename tile_shape_in::DType>::TypeCode),
          "i"(type_traits<typename tile_shape_out::DType>::TypeCode),
          "Tr"(src.data()),
          "i"(tile_type_traits<typename tile_shape_out::TileDType>::TilesizeCode),
          "r"(valid_col), "r"(valid_row), "i"(tile_shape_out::Cols));
}

template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          bool UseMx,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int XDim = 1, int YDim = 1,
          int scaleD = qD>
void flash_attention_2d_unroll_shared_impl(
    vector_dtype *out_ptr, matrix_dtype *q_ptr, matrix_dtype *k_ptr,
    matrix_dtype *v_ptr, uint8_t *q_scale_ptr, uint8_t *k_scale_ptr,
    uint8_t *v_scale_ptr,
    matrix_dtype *prob_scratch, uint8_t *prob_scale_scratch) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredVD = vD;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kQScaleCols = (kStoredQD + 31) / 32;
    constexpr int kTkScaleRows = (kStoredTk + 31) / 32;
    constexpr int kVectorStateRows = 32;
    constexpr int kSharedKRowBytes = 1024;
    constexpr int kMaxStoredKChunk =
        kSharedKRowBytes / sizeof(matrix_dtype);
    constexpr int kQKStoredChunk =
        kStoredQD < kMaxStoredKChunk ? kStoredQD : kMaxStoredKChunk;
    constexpr int kPVStoredChunk =
        kStoredTk < kMaxStoredKChunk ? kStoredTk : kMaxStoredKChunk;
    constexpr int kQKBlocks = kStoredQD / kQKStoredChunk;
    constexpr int kPVBlocks = kStoredTk / kPVStoredChunk;
    constexpr int kQKScaleCols = (kQKStoredChunk + 31) / 32;
    constexpr int kPVScaleRows = (kPVStoredChunk + 31) / 32;
    static_assert(XDim > 0 && YDim > 0,
                  "X_dim and Y_dim must both be positive");
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8+) or 2 (packed FP4x2)");
    static_assert(qD % PackedFactor == 0 && kTk % PackedFactor == 0,
                  "logical matrix dimensions must be divisible by PackedFactor");
    static_assert(kStoredQD % kQKStoredChunk == 0 &&
                      kStoredTk % kPVStoredChunk == 0,
                  "QK and PV K dimensions must be divisible by their shared chunks");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");

    using gmQ = global_tensor<matrix_dtype, RowMajor<Sq, kStoredQD>>;
    using gmK =
        global_tensor<matrix_dtype, ColMajor<kStoredQD, Skv>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<Skv / PackedFactor, kStoredVD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    using tileQMatrix =
        SharedMatrixLeft<matrix_dtype, kGroupM, kQKStoredChunk>;
    using tileKMatrix =
        SharedMatrixRight<matrix_dtype, kQKStoredChunk, kTk>;
    using tileVMatrix =
        SharedMatrixRight<matrix_dtype, kPVStoredChunk, kStoredVD>;
    using tilePMatrix =
        SharedMatrixLeft<matrix_dtype, kGroupM, kPVStoredChunk>;
    using tileQ = SharedTile<tileQMatrix>;
    using tileK = SharedTile<tileKMatrix>;
    using tileV = SharedTile<tileVMatrix>;
    using tileP = SharedTile<tilePMatrix>;

    using tileScoreM16 = CubeAccumulatorM16<float, kPeTm, kTk>;
    using tileScoreM32 = CubeAccumulatorM32<float, kPeTm, kTk>;
    using tileScoreCube =
        std::conditional_t<(kPeTm <= 16), tileScoreM16, tileScoreM32>;
    using tilePVM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using tilePVM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using tilePVCube =
        std::conditional_t<(kPeTm <= 16), tilePVM16, tilePVM32>;

    // QK score tile: stays in persistent CUBE M16/M32 layout throughout
    // softmax (scale, rowmax, exp, rowsum). No Vec RowMajor bridge needed.
    using tileW = tileScoreCube;

    // P shard: a local CUBE Left tile (CubeTileM16/M32). TCVT from the Acc
    // score tile to this Left tile is cross-Location (Acc->Left), same
    // layout (CubeM32), pure dtype cast. Because the shard is already a
    // CUBE Left operand in CubeM32 -- matching the TMATMUL left-operand
    // contract -- it can be fed directly to P*V TMATMUL without any
    // TSTORE_CUBE, GM round-trip, or SharedTile reload. Each PE keeps its
    // own [kPeTm, kStoredTk] P shard in tile registers and multiplies it
    // against the shared V Right tile.
    using tilePShardM16 =
        CubeTileM16<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShardM32 =
        CubeTileM32<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShard =
        std::conditional_t<(kPeTm <= 16), tilePShardM16, tilePShardM32>;
    // Fallback Acc shard for the MX path, which still needs TSTORE_CUBE.
    using tilePShardAccM16 =
        CubeAccumulatorM16<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShardAccM32 =
        CubeAccumulatorM32<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShardAcc =
        std::conditional_t<(kPeTm <= 16), tilePShardAccM16, tilePShardAccM32>;

    using tileO = tilePVCube;
    using tileOCastM16 =
        CubeAccumulatorM16<vector_dtype, kPeTm, vD>;
    using tileOCastM32 =
        CubeAccumulatorM32<vector_dtype, kPeTm, vD>;
    using tileOCast =
        std::conditional_t<(kPeTm <= 16), tileOCastM16, tileOCastM32>;

    constexpr int kVectorStateCols = 1;
    using tileMax =
        Tile<Location::Vec, float, kVectorStateRows, kVectorStateCols,
             BLayout::RowMajor, kPeTm, 1>;
    using tileSum = tileMax;
    using tileScale = tileMax;

    static_assert(!UseMx || (qD % 32 == 0 && vD % 32 == 0 && kTk % 32 == 0),
                  "MX/HiF FA requires qD, vD and kTk divisible by 32");
    using scale_dtype = __fp8_e8m0;
    using gmQScale = global_tensor<scale_dtype, RowMajor<Sq, kQScaleCols>>;
    using gmKScale = global_tensor<scale_dtype, ColMajor<kQScaleCols, Skv>>;
    using gmVScale = global_tensor<scale_dtype, RowMajor<kTkScaleRows, vD>>;
    using tileQScaleMatrix =
        SharedMatrixLeft<scale_dtype, kGroupM, kQKScaleCols>;
    using tileKScaleMatrix =
        SharedMatrixRight<scale_dtype, kQKScaleCols, kTk>;
    using tileVScaleMatrix =
        SharedMatrixRight<scale_dtype, kPVScaleRows, vD>;
    using tilePScaleMatrix =
        SharedMatrixLeft<scale_dtype, kGroupM, kPVScaleRows>;
    using tileQScale = SharedTile<tileQScaleMatrix>;
    using tileKScale = SharedTile<tileKScaleMatrix>;
    using tileVScale = SharedTile<tileVScaleMatrix>;
    using tilePScale = SharedTile<tilePScaleMatrix>;

    using itQ = global_iterator<gmQ, tileQMatrix>;
    using itK = global_iterator<gmK, tileKMatrix>;
    using itV = global_iterator<gmV, tileVMatrix>;
    using itO = global_iterator<gmO, tileOCast>;
    using itQScale = global_iterator<gmQScale, tileQScaleMatrix>;
    using itKScale = global_iterator<gmKScale, tileKScaleMatrix>;
    using itVScale = global_iterator<gmVScale, tileVScaleMatrix>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);
    itQScale gIterQScale(reinterpret_cast<scale_dtype *>(q_scale_ptr));
    itKScale gIterKScale(reinterpret_cast<scale_dtype *>(k_scale_ptr));
    itVScale gIterVScale(reinterpret_cast<scale_dtype *>(v_scale_ptr));

    // P shard GM buffer: each PE stores its kPeTm rows into this RowMajor
    // tensor. TSTORE_CUBE converts CubeM32->ND on store, so the GM buffer
    // is plain RowMajor matrix_dtype.
    using gmProbScratch =
        global_tensor<matrix_dtype, RowMajor<kGroupM, kStoredTk>>;
    using gmProbScaleScratch =
        global_tensor<scale_dtype, RowMajor<kGroupM, kTkScaleRows>>;
    using itP = global_iterator<gmProbScratch, tilePMatrix>;
    using itPScale =
        global_iterator<gmProbScaleScratch, tilePScaleMatrix>;
    gmProbScratch gProb(prob_scratch);
    gmProbScaleScratch gProbScale(
        reinterpret_cast<scale_dtype *>(prob_scale_scratch));
    using itPShard = global_iterator<gmProbScratch, tilePShard>;
    itPShard gIterProb(prob_scratch);
    itP gIterP(prob_scratch);
    itPScale gIterPScale(
        reinterpret_cast<scale_dtype *>(prob_scale_scratch));

    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kQKSharedBytes =
        XDim * (tileQMatrix::LogicalTileBytes +
                (UseMx ? tileQScaleMatrix::LogicalTileBytes : 0)) +
        YDim * (tileKMatrix::LogicalTileBytes +
                (UseMx ? tileKScaleMatrix::LogicalTileBytes : 0));
    constexpr int kPVSharedBytes =
        tilePMatrix::LogicalTileBytes + tileVMatrix::LogicalTileBytes +
        (UseMx ? tilePScaleMatrix::LogicalTileBytes +
                     tileVScaleMatrix::LogicalTileBytes
               : 0);
    static_assert(kQKSharedBytes <= kSharedTRegBytes,
                  "Q/K chunks and scales exceed the 256 KiB SharedTReg pool");
    static_assert(kPVSharedBytes <= kSharedTRegBytes,
                  "P/V chunks and scales exceed the 256 KiB SharedTReg pool");

    const float scale = 1.0f / sqrt((float)scaleD);
    constexpr int Qb = Sq / kGroupM;
    constexpr int Kb = (Skv + kTk - 1) / kTk;
    static_assert(Qb % XDim == 0,
                  "Sq/kTm must be divisible by X_dim");
    static_assert(Kb % YDim == 0,
                  "Skv/kTk must be divisible by Y_dim");

#pragma clang loop unroll(full)
    for (int i = 0; i < Qb; i += XDim) {
        tileMax tMax[XDim];
        tileSum tSum[XDim];
        tileO tO[XDim];
        tileScale tScale[XDim];
#pragma clang loop unroll(full)
        for (int x = 0; x < XDim; ++x) {
            TEXPANDS(tMax[x], -1e30f);
            TEXPANDS(tSum[x], 0.0f);
        }

#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j += YDim) {
            tileW tW[XDim][YDim];
#pragma clang loop unroll(full)
            for (int qk = 0; qk < kQKBlocks; ++qk) {
                tileQ tQ[XDim];
                tileQScale tQScale[XDim];
                tileK tK[YDim];
                tileKScale tKScale[YDim];
#pragma clang loop unroll(full)
                for (int x = 0; x < XDim; ++x) {
                    auto gQ = gIterQ(i + x, qk);
                    TLOAD<tileQMatrix, 1>(tQ[x], gQ);
                    if constexpr (UseMx) {
                        auto gQScale = gIterQScale(i + x, qk);
                        TLOAD<tileQScaleMatrix, 1>(tQScale[x], gQScale);
                    }
                }
#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    auto gK = gIterK(qk, j + y);
                    TLOAD<tileKMatrix, 1>(tK[y], gK);
                    if constexpr (UseMx) {
                        auto gKScale = gIterKScale(qk, j + y);
                        TLOAD<tileKScaleMatrix, 1>(tKScale[y], gKScale);
                    }
                }
#pragma clang loop unroll(full)
                for (int x = 0; x < XDim; ++x) {
#pragma clang loop unroll(full)
                    for (int y = 0; y < YDim; ++y) {
                        auto qkOptions = fixp::keep_acc();
                        if constexpr (UseMx) {
                            if (qk == 0) {
                                TMATMUL_MX<3>(
                                    tW[x][y], tQ[x], tQScale[x],
                                    tK[y], tKScale[y], qkOptions);
                            } else {
                                TMATMUL_MX_ACC<3>(
                                    tW[x][y], tW[x][y],
                                    tQ[x], tQScale[x], tK[y], tKScale[y],
                                    qkOptions);
                            }
                        } else {
                            if (qk == 0) {
                                TMATMUL(tW[x][y], tQ[x], tK[y],
                                        qkOptions);
                            } else {
                                TMATMUL_ACC(tW[x][y],
                                            tW[x][y], tQ[x], tK[y],
                                            qkOptions);
                            }
                        }
                    }
                }
            }
#pragma clang loop unroll(full)
            for (int x = 0; x < XDim; ++x) {
#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    TMULS(tW[x][y], tW[x][y], scale);
                }
            }

            tileMax tNewMax[XDim];
            tileSum tNewSum[XDim];
            tileMax tLocalMax[XDim][YDim];
            tileSum tLocalSum[XDim][YDim];
            tileSum tScaledOldSum[XDim];
            tileMax tMaxReduce[XDim][YDim];
            tileSum tSumReduce[XDim][YDim];

#pragma clang loop unroll(full)
            for (int x = 0; x < XDim; ++x) {
#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    TROWMAX(tLocalMax[x][y], tW[x][y]);
                    if (y == 0) {
                        TMAX(tMaxReduce[x][y], tMax[x],
                             tLocalMax[x][y]);
                    } else {
                        TMAX(tMaxReduce[x][y], tMaxReduce[x][y - 1],
                             tLocalMax[x][y]);
                    }
                }
                tNewMax[x] = tMaxReduce[x][YDim - 1];
                TSUB(tScale[x], tMax[x], tNewMax[x]);
                TEXP(tScale[x], tScale[x]);
                TMUL(tScaledOldSum[x], tSum[x], tScale[x]);

#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    TROWEXPANDSUB(tW[x][y], tW[x][y], tNewMax[x]);
                    TEXP(tW[x][y], tW[x][y]);
                    TROWSUM(tLocalSum[x][y], tW[x][y]);
                    if (y == 0) {
                        TADD(tSumReduce[x][y], tScaledOldSum[x],
                             tLocalSum[x][y]);
                    } else {
                        TADD(tSumReduce[x][y], tSumReduce[x][y - 1],
                             tLocalSum[x][y]);
                    }
                }
                tNewSum[x] = tSumReduce[x][YDim - 1];
            }

#ifndef FA_DISABLE_CUBE_TSTORE
            tileO tPV[XDim];
            // Non-MX: local-Left P stays in tile registers -- no GM round-trip.
            if constexpr (!UseMx) {
                static_assert(kPVBlocks == 1,
                              "local-Left P requires kPVBlocks==1; "
                              "increase kMaxStoredKChunk or reduce kTk");
            }
            tilePShard tPShard[XDim][YDim];
#pragma clang loop unroll(full)
            for (int x = 0; x < XDim; ++x) {
#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    if constexpr (UseMx) {
                        // MX fallback: TCVT to Acc, TSTORE_CUBE to GM
                        // (the MX path needs SharedTile P for the scale).
                        tilePShardAcc tPAcc;
                        if constexpr (PackedFactor == 2) {
                            fa_tcvt_packed_x2(tPAcc, tW[x][y]);
                        } else {
                            TCVT(tPAcc, tW[x][y]);
                        }
                        auto gProbShard = gIterProb(tid, 0);
                        TSTORE_CUBE(gProbShard, tPAcc);
                    } else {
                        // Non-MX: TCVT to local-Left CUBE shard.
                        // Cross-Location (Acc->Left), same layout
                        // (CubeM32), pure dtype cast -- no GM round-trip.
                        if constexpr (PackedFactor == 2) {
                            fa_tcvt_packed_x2(tPShard[x][y], tW[x][y]);
                        } else {
                            TCVT(tPShard[x][y], tW[x][y]);
                        }
                    }
                    tilePVCube tPVCube;
#pragma clang loop unroll(full)
                    for (int pvk = 0; pvk < kPVBlocks; ++pvk) {
                        tileV tV;
                        auto gV =
                            gIterV((j + y) * kPVBlocks + pvk, 0);
                        TLOAD<tileVMatrix, 1>(tV, gV);
                        if constexpr (UseMx) {
                            tileP tP;
                            tilePScale tPScale;
                            tileVScale tVScale;
                            auto gP = gIterP(0, pvk);
                            auto gPScale = gIterPScale(0, pvk);
                            auto gVScale = gIterVScale(
                                (j + y) * kPVBlocks + pvk, 0);
                            TLOAD<tilePMatrix, 1>(tP, gP);
                            TLOAD<tilePScaleMatrix, 1>(tPScale, gPScale);
                            TLOAD<tileVScaleMatrix, 1>(tVScale, gVScale);
                            if (pvk == 0) {
                                TMATMUL_MX<3>(
                                    tPVCube, tP, tPScale, tV, tVScale,
                                    fixp::keep_acc());
                            } else {
                                TMATMUL_MX_ACC<3>(
                                    tPVCube, tPVCube, tP, tPScale,
                                    tV, tVScale, fixp::keep_acc());
                            }
                        } else {
                            // Local-Left P + Shared-Right V: no GM
                            // round-trip. tPShard is a CubeTileM32
                            // (Location::Left, CubeM32) used directly as
                            // the TMATMUL left operand against the shared
                            // V Right tile.
                            if (pvk == 0) {
                                TMATMUL(tPVCube, tPShard[x][y], tV);
                            } else {
                                TMATMUL_ACC(tPVCube, tPVCube,
                                            tPShard[x][y], tV);
                            }
                        }
                    }
                    if (y == 0) {
                        tPV[x] = tPVCube;
                    } else {
                        TADD(tPV[x], tPV[x], tPVCube);
                    }
                }
                if (j == 0) {
                    tO[x] = tPV[x];
                } else {
                    TROWEXPANDMUL(tO[x], tO[x], tScale[x]);
                    TADD(tO[x], tO[x], tPV[x]);
                }
                tMax[x] = tNewMax[x];
                tSum[x] = tNewSum[x];
            }
#endif
        }

#ifndef FA_DISABLE_CUBE_TSTORE
        tileSum tInvSum[XDim];
        tileOCast tOCast[XDim];
#pragma clang loop unroll(full)
        for (int x = 0; x < XDim; ++x) {
            TRECIP(tInvSum[x], tSum[x]);
            TROWEXPANDMUL(tO[x], tO[x], tInvSum[x]);
            TCVT(tOCast[x], tO[x]);
            auto dstO =
                gIterO((i + x) * kPeNum + tid, 0);
            TSTORE_CUBE(dstO, tOCast[x]);
        }
#endif
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_tmatmul_pto(dtype *out_ptr, dtype *q_ptr,
                                           dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        dtype, dtype, 1, false, Sq, Skv, qD, vD, kTm, kTk, 1, 1, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr, nullptr, nullptr, nullptr,
        nullptr, nullptr);
}
