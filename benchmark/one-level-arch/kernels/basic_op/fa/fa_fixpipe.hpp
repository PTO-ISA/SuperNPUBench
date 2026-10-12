#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// 4-PE cooperative TMATMUL FlashAttention — fixpipe version.
//
// O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// This kernel follows fa_2d_unroll_gmma.hpp exactly, except that QK's
// standalone TROWMAX is fused into the QK TMATMUL through fixpipe RowMaxEn.
// The fixpipe result is the row maximum of the unscaled FP32 accumulator, so
// it is scaled by the same positive 1/sqrt(scaleD) factor as the score tile.
// TROWSUM remains a TileOp because B.FPATR has no row-sum output.
//
// Shared B declares its physical RowMajor shape: without transpose_b it is
// [N, K], while with transpose_b it is [K, N]. K is loaded in its natural
// [kTk, qD] = [N, K] order for QK^T, so QK does not set transpose_b.
// V is loaded in its natural [kTk, vD] = [K, N] order, so PV does set it.

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
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_fixpipe_impl(
    vector_dtype *out_ptr, matrix_dtype *q_ptr, matrix_dtype *k_ptr,
    matrix_dtype *v_ptr) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredSkv = Skv / PackedFactor;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kTileRows = (kGroupM <= 64) ? 64 : 128;
    constexpr int kPVStoredChunk = kStoredTk;
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8+) or 2 (packed FP4x2)");
    static_assert(qD % PackedFactor == 0 && kTk % PackedFactor == 0,
                  "logical matrix dimensions must be divisible by PackedFactor");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");
    static_assert(kPeTm == 32,
                  "fixpipe RowMaxOut currently requires the 32-row CUBE_M32 "
                  "PE shape; the API cannot represent a 16-row FP32 "
                  "CUBE_M16 one-column output in the minimum 128 bytes");

    // GM tensors are RowMajor in their natural [rows, cols] orientation.
    // K is [Skv, qD] with the head dimension contiguous.
    using gmQ = global_tensor<matrix_dtype, RowMajor<Sq, kStoredQD>>;
    using gmK = global_tensor<matrix_dtype, RowMajor<Skv, kStoredQD>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<kStoredSkv, vD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    using tileQMatrix =
        SharedMatrixLeft<matrix_dtype, kTileRows, kStoredQD,
                         kGroupM, kStoredQD>;
    // K tile is physically [N=kTk, K=qD/PackedFactor]. Shared B without
    // transpose_b interprets this as the right operand of QK^T.
    using tileKMatrix =
        SharedMatrixRight<matrix_dtype, kTk, kStoredQD>;
    using tileVMatrix =
        SharedMatrixRight<matrix_dtype, kPVStoredChunk, vD>;
    using tileQ = SharedTile<tileQMatrix>;
    using tileK = SharedTile<tileKMatrix>;
    using tileV = SharedTile<tileVMatrix>;

    // Match fa_2d_unroll_gmma: QK writes directly to the vector-precision
    // Left tile.  RowMax is still computed from the FP32 FullAcc before this
    // optional fixpipe conversion.
    using tileWM16 = CubeTileM16<vector_dtype, kPeTm, kTk>;
    using tileWM32 = CubeTileM32<vector_dtype, kPeTm, kTk>;
    using tileW = std::conditional_t<(kPeTm <= 16), tileWM16, tileWM32>;

    using tilePVM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using tilePVM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using tilePVCube =
        std::conditional_t<(kPeTm <= 16), tilePVM16, tilePVM32>;
    using tileO = tilePVCube;

    using tilePShardM16 = CubeTileM16<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShardM32 = CubeTileM32<matrix_dtype, kPeTm, kStoredTk>;
    using tilePShard =
        std::conditional_t<(kPeTm <= 16), tilePShardM16, tilePShardM32>;

    using tileOCastM16 = CubeAccumulatorM16<vector_dtype, kPeTm, vD>;
    using tileOCastM32 = CubeAccumulatorM32<vector_dtype, kPeTm, vD>;
    using tileOCast =
        std::conditional_t<(kPeTm <= 16), tileOCastM16, tileOCastM32>;

    // TROWSUM keeps the same wide reduction carrier as fa_2d_unroll_gmma.
    using tileReduceM16 = VecTileM16<vector_dtype, kPeTm, kTk, kPeTm, 1>;
    using tileReduceM32 = VecTileM32<vector_dtype, kPeTm, kTk, kPeTm, 1>;
    using tileReduce =
        std::conditional_t<(kPeTm <= 16), tileReduceM16, tileReduceM32>;

    using tileMaxM16 = VecTileM16<vector_dtype, kPeTm, 1, kPeTm, 1>;
    using tileMaxM32 = VecTileM32<vector_dtype, kPeTm, 1, kPeTm, 1>;
    using tileMax =
        std::conditional_t<(kPeTm <= 16), tileMaxM16, tileMaxM32>;
    using tileScale = tileMax;
    using tileRowFp32M16 = VecTileM16<float, kPeTm, 1, kPeTm, 1>;
    using tileRowFp32M32 = VecTileM32<float, kPeTm, 1, kPeTm, 1>;
    using tileRowFp32 =
        std::conditional_t<(kPeTm <= 16), tileRowFp32M16, tileRowFp32M32>;
    using tileLocalSum = tileMax;
    using tileSum = tileRowFp32;

    // Fixpipe auxiliary outputs must be FP32, physical one-column carriers,
    // and at least 128 bytes. The supported CUBE_M32 shape is exactly 128B.
    using tileFixpipeRowMax = VecTileM32<float, 32, 1, kPeTm, 1>;

    using itQ = global_iterator<gmQ, tileQMatrix>;
    using itK = global_iterator<gmK, tileKMatrix>;
    using itV = global_iterator<gmV, tileVMatrix>;
    using itO = global_iterator<gmO, tileOCast>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kQKSharedBytes =
        tileQMatrix::LogicalTileBytes + tileKMatrix::LogicalTileBytes;
    constexpr int kPVSharedBytes = tileVMatrix::LogicalTileBytes;
    static_assert(kQKSharedBytes <= kSharedTRegBytes,
                  "Q/K chunks exceed the 256 KiB SharedTReg pool");
    static_assert(kPVSharedBytes <= kSharedTRegBytes,
                  "V chunk exceeds the 256 KiB SharedTReg pool");

    const float scale = 1.0f / sqrt((float)scaleD);
    constexpr int Qb = Sq / kGroupM;
    constexpr int Kb = (Skv + kTk - 1) / kTk;

    // PV keeps the natural [K,N] V tile and therefore selects transpose_b.
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int i = 0; i < Qb; ++i) {
        tileMax tMax;
        tileSum tSum;
        tileO tO;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        // Q is loop-invariant w.r.t. Kb — load once per Q block.
        tileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD<tileQMatrix, 1>(tQ, gQ);

#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            tileW tW;

            // --- QK matmul with fused row_max (fixpipe) ---
            tileK tK;
            auto gK = gIterK(j, 0);
            TLOAD<tileKMatrix, 1>(tK, gK);

            // RowMaxOut is an FP32 auxiliary destination computed from the
            // unscaled FullAcc before conversion. K is already stored as
            // [N,K], so QK must not request transpose_b.
            tileFixpipeRowMax tLocalMaxFp32;
            if constexpr (std::is_same_v<vector_dtype, float>) {
                auto qkOptions = fixp::keep_acc().row_max(tLocalMaxFp32);
                TMATMUL(tW, tQ, tK, qkOptions);
            } else {
                static_assert(std::is_same_v<vector_dtype, __bf16>,
                              "QK fixpipe output supports FP32 or BF16 vector dtype");
                auto qkOptions = fixp::bf16().row_max(tLocalMaxFp32);
                TMATMUL(tW, tQ, tK, qkOptions);
            }

            // Scale both score and row_max (row_max output is unscaled).
            // max(c*x) = c*max(x) for c > 0, so scaling tLocalMax after
            // the fused row_max is equivalent to TROWMAX on the scaled tile.
            TMULS(tW, tW, scale);
            TMULS(tLocalMaxFp32, tLocalMaxFp32, scale);

            tileMax tLocalMax;
            if constexpr (std::is_same_v<tileMax, tileFixpipeRowMax>) {
                tLocalMax = tLocalMaxFp32;
            } else {
                TCVT(tLocalMax, tLocalMaxFp32);
            }

            // --- Softmax ---
            // TROWMAX is eliminated — tLocalMax was produced by the
            // fused row_max in the QK TMATMUL above.
            tileMax tNewMax;
            tileScale tScale;
            tileRowFp32 tScaleFp32;
            TMAX(tNewMax, tMax, tLocalMax);
            if (j != 0) {
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
                if constexpr (std::is_same_v<vector_dtype, float>) {
                    TROWEXPANDMUL(tO, tO, tScale);
                } else {
                    TCVT(tScaleFp32, tScale);
                    TROWEXPANDMUL(tO, tO, tScaleFp32);
                }
            }

            // Exponentiate current scores (common to both branches)
            TROWEXPANDEXPDIF(tW, tW, tNewMax);

            tileReduce tLocalSumR;
            TROWSUM(tLocalSumR, tW);

            tileSum tNewSum;
            if constexpr (std::is_same_v<vector_dtype, float>) {
                auto tLocalSum = TREDUCEPREFIXVIEW<tileSum>(tLocalSumR);
                if (j == 0) {
                    TADD(tNewSum, tSum, tLocalSum);
                } else {
                    TFMA(tNewSum, tSum, tScale, tLocalSum);
                }
            } else {
                auto tLocalSumView =
                    TREDUCEPREFIXVIEW<tileLocalSum>(tLocalSumR);
                tileLocalSum tLocalSumZero;
                tileLocalSum tLocalSumCompact;
                TEXPANDS(tLocalSumZero, 0.0f);
                TADD(tLocalSumCompact, tLocalSumZero, tLocalSumView);
                tileSum tLocalSumFp32;
                TCVT(tLocalSumFp32, tLocalSumCompact);
                if (j == 0) {
                    TADD(tNewSum, tSum, tLocalSumFp32);
                } else {
                    TFMA(tNewSum, tSum, tScaleFp32, tLocalSumFp32);
                }
            }

            // --- PV matmul ---
            tileV tV;
            auto gV = gIterV(j, 0);
            TLOAD<tileVMatrix, 1>(tV, gV);

            if constexpr (PackedFactor == 1 &&
                          std::is_same_v<matrix_dtype, vector_dtype>) {
                // The softmax probability already matches the configured
                // CUBE input dtype, so it can feed PV directly.
                if (j == 0) {
                    TMATMUL(tO, tW, tV, pvOptions, kGroupM);
                } else {
                    TMATMUL_ACC(tO, tO, tW, tV, pvOptions, kGroupM);
                }
            } else {
                // Non-FP32 or packed: TCVT to local-Left CUBE shard
                tilePShard tPShard;
                if constexpr (PackedFactor == 2) {
                    fa_tcvt_packed_x2(tPShard, tW);
                } else {
                    TCVT(tPShard, tW);
                }
                if (j == 0) {
                    TMATMUL(tO, tPShard, tV, pvOptions, kGroupM);
                } else {
                    TMATMUL_ACC(tO, tO, tPShard, tV, pvOptions, kGroupM);
                }
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        TROWEXPANDDIV(tO, tO, tSum);
        auto dstO = gIterO(i * kPeNum + tid, 0);
        if constexpr (std::is_same_v<vector_dtype, float>) {
            TSTORE(dstO, tO);
        } else {
            tileOCast tOCast;
            TCVT(tOCast, tO);
            TSTORE(dstO, tOCast);
        }
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_fixpipe_pto(dtype *out_ptr, dtype *q_ptr,
                                  dtype *k_ptr, dtype *v_ptr) {
    flash_attention_fixpipe_impl<
        dtype, dtype, 1, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}
