#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// 4-PE cooperative TMATMUL FlashAttention — subview version.
//
// O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// This variant is identical to fa_2d_unroll_gmma.hpp except for the reduction ops
// (TROWMAX, TROWSUM).  The ISA limits reduction-source tiles to
// 2048 bytes; when the QK result tile tW exceeds that capacity, the
// reduction is performed on TPARTVIEW column sub-views and the partial
// results are combined with TMAX (for row-max) or TADD (for row-sum).
// All other tileops (TMATMUL, TCVT, expand ops, TLOAD/TSTORE) have no
// 2048-byte constraint and operate on the full tile as in fa_2d_unroll_gmma.
//
// Inherits all optimisations from fa_2d_unroll_gmma v2:
//   - Left matrix M physical rows are 128 or 64; ValidRow = actual kGroupM.
//   - kSharedKRowBytes removed; SharedTile total ≤ 256 KB.
//   - Vector state tiles use VecTileM32 (CubeM32 layout) with vector_dtype.
//   - j==0: tNewMax = tLocalMax directly (skip TMAX).
//   - TSUB+TEXP replaced by TROWEXPANDEXPDIF (fused exp(a-b)) in j!=0 branch.
//   - tW is CubeTileM32 (Left) instead of CubeAccumulatorM32 (Acc).
//   - TRECIP+TROWEXPANDMUL replaced by TROWEXPANDDIV (fused division).
//   - PV j==0 TMATMUL uses Options+groupM overload.

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

// ── Partitioned reduction helpers ──────────────────────────────────
// The ISA requires reduction-source tiles (TROWMAX, TROWSUM, …) to have
// an allocated capacity ≤ 2048 bytes.  When the QK result tile tW is
// larger than that, the reduction is split across column sub-views
// created by TPARTVIEW and the partial results are combined pairwise.

// Partitioned row-max: TROWMAX on each column sub-view, TMAX to combine.
template <int Parts, typename OutTile, typename SubTileType,
          typename ParentTile>
inline void fa_subview_row_max(OutTile &out, ParentTile &parent) {
    if constexpr (Parts <= 1) {
        TROWMAX(out, parent);
    } else {
        auto parts = TPARTVIEW<SubTileType, 1, Parts>(parent);
        OutTile partial;
        auto part0 = parts[0][0];
        TROWMAX(partial, part0);
        out = partial;
#pragma clang loop unroll(full)
        for (int p = 1; p < Parts; ++p) {
            auto part = parts[0][p];
            TROWMAX(partial, part);
            OutTile combined;
            TMAX(combined, out, partial);
            out = combined;
        }
    }
}

// Partitioned row-sum: TROWSUM on each column sub-view, TADD to combine.
template <int Parts, typename OutTile, typename SubTileType,
          typename ParentTile>
inline void fa_subview_row_sum(OutTile &out, ParentTile &parent) {
    if constexpr (Parts <= 1) {
        TROWSUM(out, parent);
    } else {
        auto parts = TPARTVIEW<SubTileType, 1, Parts>(parent);
        OutTile partial;
        auto part0 = parts[0][0];
        TROWSUM(partial, part0);
        out = partial;
#pragma clang loop unroll(full)
        for (int p = 1; p < Parts; ++p) {
            auto part = parts[0][p];
            TROWSUM(partial, part);
            OutTile combined;
            TADD(combined, out, partial);
            out = combined;
        }
    }
}

template <typename matrix_dtype, typename vector_dtype, int PackedFactor,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_shared_impl(
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

    // GM tensors use their natural RowMajor shapes:
    // Q [Sq, qD], K [Skv, qD], V [Skv, vD], O [Sq, vD].
    using gmQ = global_tensor<matrix_dtype, RowMajor<Sq, kStoredQD>>;
    using gmK = global_tensor<matrix_dtype, RowMajor<Skv, kStoredQD>>;
    using gmV = global_tensor<matrix_dtype, RowMajor<kStoredSkv, vD>>;
    using gmO = global_tensor<vector_dtype, RowMajor<Sq, vD>>;

    using tileQMatrix =
        SharedMatrixLeft<matrix_dtype, kTileRows, kQKStoredChunk,
                         kGroupM, kQKStoredChunk>;
    // SharedTReg operands are also RowMajor. K is loaded as [Tk, qD] and
    // transposed by the QK TMATMUL option to form the effective [qD, Tk] B.
    using tileKMatrix =
        SharedMatrixRight<matrix_dtype, kTk, kQKStoredChunk>;
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

    // ── Reduction sub-view parameters ───────────────────────────────
    // The ISA limits reduction-source tiles to 2048 bytes.  Compute the
    // minimum number of column partitions so each sub-tile fits.
    constexpr int kMaxReduceBytes = 2048;
    constexpr int kTileWBytes = tileW::LogicalTileBytes;
    constexpr int kReduceParts =
        (kTileWBytes + kMaxReduceBytes - 1) / kMaxReduceBytes;
    static_assert(kReduceParts >= 1, "at least one reduction part");
    static_assert(kTk % kReduceParts == 0,
                  "kTk must be divisible by the number of reduction parts");
    constexpr int kReduceSubCols = kTk / kReduceParts;
    using tileWSubM16 = CubeTileM16<vector_dtype, kPeTm, kReduceSubCols>;
    using tileWSubM32 = CubeTileM32<vector_dtype, kPeTm, kReduceSubCols>;
    using tileWSub =
        std::conditional_t<(kPeTm <= 16), tileWSubM16, tileWSubM32>;
    static_assert(tileWSub::LogicalTileBytes <= kMaxReduceBytes,
                  "Reduction sub-tile must fit within the 2KB ISA limit");

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

#pragma clang loop unroll(full)
    for (int i = 0; i < Qb; ++i) {
        tileO tO;
        tileMax tMax;
        tileSum tSum;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);
        auto gQ = gIterQ(i, 0);
        tileQ tQ;
        TLOAD<tileQMatrix, 1>(tQ, gQ);

#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            tileW tW;

            // --- QK matmul ---
            tileK tK;
            auto gK = gIterK(j, 0);
            TLOAD<tileKMatrix, 1>(tK, gK);

            auto qkOptions = fixp::keep_acc().transpose_b();
            TMATMUL(tW, tQ, tK, qkOptions);

            // Scale
            TMULS(tW, tW, scale);

            // --- Softmax ---
            tileMax tLocalMax;
            // Sub-view row-max: partition tW by columns when it exceeds
            // the 2048-byte reduction-source ISA limit.
            fa_subview_row_max<kReduceParts, tileMax, tileWSub>(
                tLocalMax, tW);

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
            // Sub-view row-sum: same partitioning as row-max above, but
            // on the exponentiated tW.  Views are recreated after the
            // in-place TROWEXPANDEXPDIF modifies tW.
            fa_subview_row_sum<kReduceParts, tileSum, tileWSub>(
                tLocalSum, tW);

            tileSum tNewSum;
            if (j == 0) {
                tNewSum = tLocalSum;
            } else {
                TFMA(tNewSum, tSum, tScale, tLocalSum);
            }

#ifndef FA_DISABLE_CUBE_TSTORE
            // --- PV matmul ---
            tileV tV;
            auto gV = gIterV(j, 0);
            TLOAD<tileVMatrix, 1>(tV, gV);

            auto pvOptions = fixp::keep_acc();
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
#endif
        }

#ifndef FA_DISABLE_CUBE_TSTORE
        TROWEXPANDDIV(tO, tO, tSum);
        auto dstO = gIterO(i * kPeNum + tid, 0);
        if constexpr (std::is_same_v<vector_dtype, float>) {
            TSTORE_CUBE(dstO, tO);
        } else {
            tileOCast tOCast;
            TCVT(tOCast, tO);
            TSTORE_CUBE(dstO, tOCast);
        }
#endif
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_tmatmul_pto(dtype *out_ptr, dtype *q_ptr,
                                           dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        dtype, dtype, 1, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}
