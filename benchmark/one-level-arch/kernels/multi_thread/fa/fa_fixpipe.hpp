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
// This variant is identical to fa_2d_unroll_gmma.hpp except for the
// row-max computation.  The fixpipe post-processing feature (B.FPATR
// RowMaxEn) fuses TROWMAX into the QK TMATMUL: the TMATMUL emits the
// row-wise maximum of the score accumulator alongside the matmul
// result, writing it to a Vec tile that fa_2d_unroll_gmma produces
// with a standalone TROWMAX tile op.
//
// The row-max output covers the *unscaled* score tile, so a companion
// TMULS scales tLocalMax by 1/sqrt(scaleD) alongside the existing
// TMULS on tW.  The remaining softmax ops (TMAX merge, TROWEXPANDEXPDIF,
// TROWSUM, TFMA, TROWEXPANDDIV) are unchanged.
//
// Inherits all optimisations from fa_2d_unroll_gmma v2:
//   - Left matrix M physical rows are 128 or 64; ValidRow = actual kGroupM.
//   - kSharedKRowBytes removed; SharedTile total ≤ 256 KB.
//   - Vector state tiles use VecTileM32 (CubeM32 layout) with vector_dtype.
//   - j==0: tNewMax = tLocalMax directly (skip TMAX).
//   - TSUB+TEXP replaced by TROWEXPANDEXPDIF (fused exp(a-b)) in j!=0 branch.
//   - tW is CubeTileM32 (Left) instead of CubeAccumulatorM32 (Acc).
//   - TRECIP+TROWEXPANDMUL replaced by TROWEXPANDDIV (fused division).
//   - PV uses transpose_b() and the 6-arg Options+groupM TMATMUL_ACC.
//   - Loop-invariant pvOptions hoisted outside both loops; Q load
//     outside Kb loop.  qkOptions stays in the Kb loop (row_max binding).

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
    constexpr int kQKStoredChunk = kStoredQD;
    constexpr int kPVStoredChunk = kStoredTk;
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be 1 (FP8+) or 2 (packed FP4x2)");
    static_assert(qD % PackedFactor == 0 && kTk % PackedFactor == 0,
                  "logical matrix dimensions must be divisible by PackedFactor");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in the range 1..128");

    // GM tensors. Q/V/O are RowMajor in their natural [rows, cols] orientation.
    // K is physically [Skv, qD] row-major (head dim contiguous), described as
    // RowMajor<Skv, qD> so the CUBE reads K^T without a transpose_b flag.
    using gmQ = global_tensor<matrix_dtype, RowMajor<Sq, kStoredQD>>;
    using gmK = global_tensor<matrix_dtype, RowMajor<Skv, kStoredQD>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<kStoredSkv, vD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    using tileQMatrix =
        SharedMatrixLeft<matrix_dtype, kTileRows, kQKStoredChunk,
                         kGroupM, kQKStoredChunk>;
    using tileKMatrix =
        SharedMatrixRight<matrix_dtype, kQKStoredChunk, kTk>;
    using tileVMatrix =
        SharedMatrixRight<matrix_dtype, kPVStoredChunk, vD>;
    using tileQ = SharedTile<tileQMatrix>;
    using tileK = SharedTile<tileKMatrix>;
    using tileV = SharedTile<tileVMatrix>;

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

    using tileMax = VecTileM32<vector_dtype, 32, 1, kPeTm, 1>;
    using tileSum = tileMax;
    using tileScale = tileMax;

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

    // pvOptions is loop-invariant — hoisted outside both loops.  qkOptions
    // stays inside the Kb loop because row_max binds a per-iteration tile.
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
            auto gK = gIterK(0, j);
            TLOAD<tileKMatrix, 1>(tK, gK);

            // fixpipe: fuse TROWMAX into QK TMATMUL via the row_max
            // post-processing feature (B.FPATR RowMaxEn).  tLocalMax
            // receives the row-wise max of the UNSCALED accumulator.
            tileMax tLocalMax;
            auto qkOptions = fixp::keep_acc().row_max(tLocalMax);
            TMATMUL(tW, tQ, tK, qkOptions);

            // Scale both score and row_max (row_max output is unscaled).
            // max(c*x) = c*max(x) for c > 0, so scaling tLocalMax after
            // the fused row_max is equivalent to TROWMAX on the scaled tile.
            TMULS(tW, tW, scale);
            TMULS(tLocalMax, tLocalMax, scale);

            // --- Softmax ---
            // TROWMAX is eliminated — tLocalMax was produced by the
            // fused row_max in the QK TMATMUL above.
            tileMax tNewMax;
            tileScale tScale;
            if (j == 0) {
                tNewMax = tLocalMax;
            } else {
                TMAX(tNewMax, tMax, tLocalMax);
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
                TROWEXPANDMUL(tO, tO, tScale);
            }

            // Exponentiate current scores (common to both branches)
            TROWEXPANDEXPDIF(tW, tW, tNewMax);

            tileSum tLocalSum;
            TROWSUM(tLocalSum, tW);

            tileSum tNewSum;
            if (j == 0) {
                tNewSum = tLocalSum;
            } else {
                TFMA(tNewSum, tSum, tScale, tLocalSum);
            }

            // --- PV matmul ---
            tileV tV;
            auto gV = gIterV(j, 0);
            TLOAD<tileVMatrix, 1>(tV, gV);

            if constexpr (PackedFactor == 1 &&
                          std::is_same_v<matrix_dtype, float>) {
                // FP32: tW is already Left, float — no TCVT
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
            TSTORE_CUBE(dstO, tO);
        } else {
            tileOCast tOCast;
            TCVT(tOCast, tO);
            TSTORE_CUBE(dstO, tOCast);
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
