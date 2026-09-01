#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// Convert two logical scalar columns into one packed-x2 cube element. The
// regular PTO TCVT wrapper requires identical physical shapes, which is right
// for scalar dtypes but cannot describe [M,K] scalar -> [M,K/2] packed input.
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

// The tileop API exposes shared TMATMUL operands as SharedTile wrapping a
// TileLeft/TileRight local shape. Q/K/V are all staged into shared tile
// storage via TLOAD, matching kernels/matmul/matmul_shared.hpp.
// 遗留
// 1.高性能上是否存在表达问题？例如，软件pingping流水是否需要暴露(性能)
// 2.layout转换是否需要对程序员可见，数据类型cube-vec之间layout转换

// 4-PE tmatmul FlashAttention programming model.
//
// This kernel models a Blackwell-like execution style where get_thread_idx()
// selects the current PE's output row range. Q/K/V are loaded as full shared
// tiles. The first group TMATMUL consumes Q_shared [kTm,qD] and produces one
// PE-local score slice [kTm/4,kTk] on each PE.
//
// Mathematical semantics:
//   O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// Big-tile vs small-tile naming:
//   - Big tile is the logical tile visible to the collective tmatmul:
//       Q_big: [kTm, qD]
//       K_big: [kTk, qD], consumed by tmatmul as K_big^T [qD, kTk]
//       W_big: [kTm, kTk]
//       V_big: [kTk, vD]
//       O_big/PV_big: [kTm, vD]
//   - Small tile is the PE-local storage unit:
//       W_pe: [kTm/4, kTk]
//       O_pe/PV_pe: [kTm/4, vD]
//   - Q/K/V are shared staging tiles, matching the
//     matmul_shared pattern where both TMATMUL operands live in SharedTile:
//       Q_shared: [kTm, qD]
//       K_shared: [kTk, qD]
//       V_shared: [kTk, vD]
//   - TLOAD uses PEMask=1 so only PE0 issues each shared-tile load. Group
//     TMATMUL maps contiguous kTm/4-row score/output slices to PE0..PE3,
//     matching matmul_shared's C path.
//
// Memory/layout contract:
//   - TLOAD/TSTORE are pure ND DMA copies. They do not transpose, swizzle, or
//     pad data while moving it between global memory and tile storage.
//   - Q, K, V, O global tensors are all RowMajor.
//   - K/V are not split by PE. They are direct row-major shared tiles. TMATMUL
//     consumes K as K^T internally; that interpretation is carried by compute,
//     not by TLOAD.
//   - O is stored as row-major [Sq, vD].
//
// Compute contract:
//   - Each PE independently executes vector tileOPs for its [kTm, *] slice.
//   - Every physical PE-local/shared tile is constrained to at most 8 KiB.
//   - tmatmul is a compiler intrinsic used as a scalar instruction in each PE's
//     program. Each PE passes only its own lhs/acc tile, while K/V are shared
//     staging tiles. The collective execution fuses the PE-local slices into
//     one logical GEMM.
//
// Current simplification:
//   - This TMATMUL example fixes one Q big tile and one K/V big tile per loop
//     step. It intentionally omits the extra array dimensions and merge logic
//     used by multi-block unrolling.

template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          bool UseMx,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int XDim = 1, int YDim = 1,
          int scaleD = qD>
void flash_attention_2d_unroll_shared_impl(
    vector_dtype *out_ptr, matrix_dtype *q_ptr, matrix_dtype *k_ptr,
    matrix_dtype *v_ptr, uint8_t *q_scale_ptr, uint8_t *k_scale_ptr,
    uint8_t *v_scale_ptr, float *score_scratch,
    matrix_dtype *prob_scratch, uint8_t *prob_scale_scratch,
    float *pv_scratch) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredVD = vD;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kQScaleCols = (kStoredQD + 31) / 32;
    constexpr int kTkScaleRows = (kStoredTk + 31) / 32;
    constexpr int kVectorStateRows = 64; // gfrun/isa not define how to fill 128B m32 with 64B half data
    // SharedTReg is a single 256 KiB pool, rather than a per-operand limit.
    // Match the shared matmul kernels by splitting both CUBE reductions along
    // K when a configured reduction is wider than the legal active operand
    // pair. Capacity configurations select Tm/Tk so a 1024-byte reduction
    // payload fits directly; MX variants use a smaller Tk to leave room for
    // their scale tiles.
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

    // The current TileOP surface provides one 256 KiB SharedTReg pool. Q/K
    // and P/V are materialized in reduction chunks below; the four-PE group
    // row split remains implicit. Capacity configurations still use 64 KiB
    // PE-local FP32 score/output tiles.

    // Global tensor layout. K's physical [Skv,K] row-major buffer is described
    // as the equivalent logical [K,Skv] ColMajor matrix. TLOAD materializes
    // each logical [K,Tk] right operand directly, matching shared matmul and
    // avoiding a B.FPATR transpose operation.
    //   gmQ: [Sq,  qD], row stride qD
    //   gmK: logical [qD, Skv], ColMajor; physical buffer [Skv, qD]
    //   gmV: [Skv, vD], row stride vD
    //   gmO: [Sq,  vD], row stride vD
    using gmQ = global_tensor<matrix_dtype, RowMajor<Sq, kStoredQD>>;
    using gmK =
        global_tensor<matrix_dtype, ColMajor<kStoredQD, Skv>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<Skv / PackedFactor, kStoredVD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    // Cooperative CUBE primaries are ordinary ND Shared rectangles. The
    // Left/Right role describes operand ordering; no Local NZ/ZN layout is
    // attached to Shared storage. K is already defined and loaded as the
    // mathematical [K,Tk] right operand, so TMATMUL needs no transpose flag.
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

    // QK score tiles:
    //   tmatmul input in each PE:
    //     tQ          -> shared Q_big [kTm, qD]
    //     tK          -> shared K tile [kTk, qD]
    //   tmatmul output in each PE:
    //     tW          -> current PE's W_pe [kPeTm, kTk], ordinary Vec tile
    //   logical collective output:
    //     W_big       -> concat W_pe from PE0..PE3, shape [kTm, kTk].
    //
    // tileW/tileWCast are PE-local vector tiles used by online softmax.
    // The group score C has only kPeTm rows. RowMajor keeps the 4-row FP32
    // slice legal because the contiguous kTk columns provide 32-byte alignment.
    using tileWAcc =
        Tile<Location::Vec, float, kPeTm, kTk, BLayout::RowMajor>;
    using tileW =
        Tile<Location::Vec, vector_dtype, kPeTm, kTk, BLayout::RowMajor>;
    using tileWCast = tileW;
    // P is produced in vector_dtype by softmax, then explicitly converted to
    // the configured cube input dtype before P*V. Packed FP4x2 uses K/2
    // physical elements, each containing two logical probability values.
    using tilePShard =
        Tile<Location::Vec, matrix_dtype, kPeTm, kStoredTk,
             BLayout::RowMajor>;

    // PV/output tiles:
    //   TMATMUL(P_pe, V_shared) -> PV_pe [kPeTm, vD]
    //   current PE receives PV_pe/tileO [kPeTm, vD].
    //   tileO accumulates the online-softmax numerator for this PE row slice.
    //   tileOCast is the dtype tile stored to gmO.
    // P is PE-local while V is shared, so this is the local-A/shared-B
    // TMATMUL form. Its local C has the same row count as the local A:
    // [kPeTm,vD]. No full-kTm physical output container is needed.
    using tileOAcc =
        Tile<Location::Vec, float, kPeTm, vD, BLayout::RowMajor>;
    using tileO =
        Tile<Location::Vec, vector_dtype, kPeTm, vD, BLayout::RowMajor>;
    using tileOCast = tileO;

    // Online softmax row-state tiles. Each PE owns kPeTm independent query
    // rows, and every row has one scalar max/sum/scale value.
    // Physical cols = 8 only for tile alignment; valid cols = 1.
    //   tileMax/tileSum/tileScale: valid shape [kPeTm, 1]
    constexpr int kVectorStateCols = 1;
    using tileMax =
        Tile<Location::Vec, vector_dtype, kVectorStateRows, kVectorStateCols,
             BLayout::RowMajor,
             kPeTm, 1>;
    using tileSum = tileMax;
    using tileScale = tileMax;
    using tileStateAcc =
        Tile<Location::Vec, float, kVectorStateRows, kVectorStateCols,
             BLayout::RowMajor, kPeTm, 1>;

    // E8M0 scale tensors used by MXFP8/MXFP4/HiF8/HiF4. Q/K/V scales are
    // supplied by global memory. The softmax probability scale is unity
    // (E8M0 encoding 127) after the explicit vector->matrix TCVT.
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
    // global_iterator describes the GM window with the underlying local tile
    // shape. TLOAD may then target either that local tile or its SharedTile
    // wrapper; SharedTile itself is intentionally not an iterator tile type.
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

    using gmScoreScratch = global_tensor<float, RowMajor<kPeTm, kTk>>;
    using gmProbScratch =
        global_tensor<matrix_dtype, RowMajor<kGroupM, kStoredTk>>;
    using gmProbScaleScratch =
        global_tensor<scale_dtype, RowMajor<kGroupM, kTkScaleRows>>;
    using gmPVScratch = global_tensor<float, RowMajor<kPeTm, vD>>;
    using itP = global_iterator<gmProbScratch, tilePMatrix>;
    using itPScale =
        global_iterator<gmProbScaleScratch, tilePScaleMatrix>;
    gmScoreScratch gScore(score_scratch);
    gmProbScratch gProb(prob_scratch);
    gmProbScaleScratch gProbScale(
        reinterpret_cast<scale_dtype *>(prob_scale_scratch));
    gmPVScratch gPV(pv_scratch);
    using itProbShard = global_iterator<gmProbScratch, tilePShard>;
    itProbShard gIterProb(prob_scratch);
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

    // Score scaling for softmax(QK / sqrt(scaleD)).
    const float scale = 1.0f / sqrt((float)scaleD);
    // Qb is counted in logical big Q tiles [kTm, qD].
    // Kb is counted in logical big K/V tiles [kTk, qD/vD].
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
            tileStateAcc tMaxInit;
            tileStateAcc tSumInit;
            TEXPANDS(tMaxInit, -1e30f);
            TEXPANDS(tSumInit, 0.0f);
            TCVT(tMax[x], tMaxInit);
            TCVT(tSum[x], tSumInit);
        }

        // A YDim group shares its K/V loads across all XDim Q tiles. The
        // group is reduced under one online-softmax max before its PV
        // contributions are accumulated.
#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; j += YDim) {
            tileScoreCube tScoreCube[XDim][YDim];
            tileWAcc tWAcc[XDim][YDim];
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
                                    tScoreCube[x][y], tQ[x], tQScale[x],
                                    tK[y], tKScale[y], qkOptions);
                            } else {
                                TMATMUL_MX_ACC<3>(
                                    tScoreCube[x][y], tScoreCube[x][y],
                                    tQ[x], tQScale[x], tK[y], tKScale[y],
                                    qkOptions);
                            }
                        } else {
                            if (qk == 0) {
                                TMATMUL(tScoreCube[x][y], tQ[x], tK[y],
                                        qkOptions);
                            } else {
                                TMATMUL_ACC(tScoreCube[x][y],
                                            tScoreCube[x][y], tQ[x], tK[y],
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
                    TSTORE_CUBE(gScore, tScoreCube[x][y]);
                    TLOAD(tWAcc[x][y], gScore);
                    TMULS(tWAcc[x][y], tWAcc[x][y], scale);
                    TCVT(tW[x][y], tWAcc[x][y]);
                }
            }

            tileMax tNewMax[XDim];
            tileSum tNewSum[XDim];
            tileMax tLocalMax[XDim][YDim];
            tileSum tLocalSum[XDim][YDim];
            tileSum tScaledOldSum[XDim];
            tileWCast tExpW[XDim][YDim];
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
                    TCVT(tExpW[x][y], tW[x][y]);
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

            tilePShard tPShard[XDim][YDim];
            tileO tPV[XDim];
#pragma clang loop unroll(full)
            for (int x = 0; x < XDim; ++x) {
#pragma clang loop unroll(full)
                for (int y = 0; y < YDim; ++y) {
                    if constexpr (PackedFactor == 2) {
                        fa_tcvt_packed_x2(tPShard[x][y], tExpW[x][y]);
                    } else {
                        TCVT(tPShard[x][y], tExpW[x][y]);
                    }
                    auto gProbShard = gIterProb(tid, 0);
                    TSTORE(gProbShard, tPShard[x][y]);
                    tilePVCube tPVCube;
#pragma clang loop unroll(full)
                    for (int pvk = 0; pvk < kPVBlocks; ++pvk) {
                        tileP tP;
                        tileV tV;
                        auto gP = gIterP(0, pvk);
                        auto gV =
                            gIterV((j + y) * kPVBlocks + pvk, 0);
                        TLOAD<tilePMatrix, 1>(tP, gP);
                        TLOAD<tileVMatrix, 1>(tV, gV);
                        if constexpr (UseMx) {
                            tilePScale tPScale;
                            tileVScale tVScale;
                            auto gPScale = gIterPScale(0, pvk);
                            auto gVScale = gIterVScale(
                                (j + y) * kPVBlocks + pvk, 0);
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
                            if (pvk == 0) {
                                TMATMUL(tPVCube, tP, tV);
                            } else {
                                TMATMUL_ACC(tPVCube, tPVCube, tP, tV);
                            }
                        }
                    }
                    TSTORE_CUBE(gPV, tPVCube);
                    tileOAcc tPVAcc;
                    TLOAD(tPVAcc, gPV);
                    tileO tPVPart;
                    TCVT(tPVPart, tPVAcc);
                    if (y == 0) {
                        tPV[x] = tPVPart;
                    } else {
                        TADD(tPV[x], tPV[x], tPVPart);
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
        }

        tileSum tInvSum[XDim];
        tileOCast tOCast[XDim];
#pragma clang loop unroll(full)
        for (int x = 0; x < XDim; ++x) {
            TRECIP(tInvSum[x], tSum[x]);
            TROWEXPANDMUL(tO[x], tO[x], tInvSum[x]);
            TCVT(tOCast[x], tO[x]);
            auto dstO =
                gIterO((i + x) * kPeNum + tid, 0);
            TSTORE(dstO, tOCast[x]);
        }
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_tmatmul_pto(dtype *out_ptr, dtype *q_ptr,
                                           dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        dtype, dtype, 1, false, Sq, Skv, qD, vD, kTm, kTk, 1, 1, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
}
