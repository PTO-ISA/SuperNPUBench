#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// Internal-accumulator K-chain phases described by PTO-ISA CCTRL[1:0].
//
//   Single : CCTRL=0, D = A * B and D is the final architectural result.
//   Begin  : CCTRL=1, start a chain and keep a raw accumulator result.
//   Middle : CCTRL=3, consume the previous internal accumulator and keep the
//            next raw accumulator result.
//   End    : CCTRL=2, consume the previous internal accumulator and publish D.
//
// The public helper intentionally has only D/A/B operands. The accumulator
// produced by a preceding phase is not repeated at the call site as an input
// operand. The current TileOP backend represents the in-place ACC form by
// forwarding D as the implicit C operand inside this adapter.
enum class TMatmulInternalAccPhase : uint8_t {
    Single,
    Begin,
    Middle,
    End,
};

template <TMatmulInternalAccPhase Phase, bool TransposeB = false>
constexpr auto fa_internal_acc_options() {
    if constexpr (Phase == TMatmulInternalAccPhase::Single) {
        if constexpr (TransposeB) {
            return fixp::keep_acc().transpose_b();
        } else {
            return fixp::keep_acc();
        }
    } else if constexpr (Phase == TMatmulInternalAccPhase::Begin) {
        // Apply transpose after CCTRL. The current compiler's CCTRL builder
        // starts from a fresh attribute object, while transpose_b is chainable.
        if constexpr (TransposeB) {
            return fixp::keep_acc().raw_acc().transpose_b();
        } else {
            return fixp::keep_acc().raw_acc();
        }
    } else if constexpr (Phase == TMatmulInternalAccPhase::Middle) {
        if constexpr (TransposeB) {
            return fixp::keep_acc().raw_acc().acc_hint().transpose_b();
        } else {
            return fixp::keep_acc().raw_acc().acc_hint();
        }
    } else {
        if constexpr (TransposeB) {
            return fixp::keep_acc().acc_hint().transpose_b();
        } else {
            return fixp::keep_acc().acc_hint();
        }
    }
}

template <TMatmulInternalAccPhase Phase, bool TransposeB = false,
          is_tile_data_v TileD, is_local_or_shared_left TileA,
          is_local_or_shared_right TileB>
inline void TMATMUL_INTERNAL_ACC(TileD &d, TileA &a, TileB &b) {
    constexpr auto options =
        fa_internal_acc_options<Phase, TransposeB>();
    if constexpr (Phase == TMatmulInternalAccPhase::Single ||
                  Phase == TMatmulInternalAccPhase::Begin) {
        TMATMUL(d, a, b, options);
    } else {
        TMATMUL_ACC(d, d, a, b, options);
    }
}

// Local-A/Shared-B cooperative form. groupM is the core-total M while D and A
// retain their per-PE row shape.
template <TMatmulInternalAccPhase Phase, bool TransposeB = false,
          is_tile_data_v TileD, is_local_or_shared_left TileA,
          is_local_or_shared_right TileB>
inline void TMATMUL_INTERNAL_ACC(TileD &d, TileA &a, TileB &b,
                                 size_t groupM) {
    constexpr auto options =
        fa_internal_acc_options<Phase, TransposeB>();
    if constexpr (Phase == TMatmulInternalAccPhase::Single ||
                  Phase == TMatmulInternalAccPhase::Begin) {
        TMATMUL(d, a, b, options, groupM);
    } else {
        TMATMUL_ACC(d, d, a, b, options, groupM);
    }
}

// Read-only column partitions alias the parent; no copy or assembly is needed.
// Each partial reduction produces [peTm, 1], combined across columns only.
template <bool IsMax, int Parts, typename SubTile, typename OutTile,
          typename ParentTile>
inline void fa_kchains_subview_reduce(OutTile &out, ParentTile &parent) {
    if constexpr (Parts == 1) {
        if constexpr (IsMax) TROWMAX(out, parent);
        else TROWSUM(out, parent);
    } else {
        auto parts = TPARTVIEW<SubTile, 1, Parts>(parent);
        auto first = parts[0][0];
        if constexpr (IsMax) TROWMAX(out, first);
        else TROWSUM(out, first);
#pragma clang loop unroll(full)
        for (int p = 1; p < Parts; ++p) {
            auto part = parts[0][p];
            OutTile partial;
            if constexpr (IsMax) TROWMAX(partial, part);
            else TROWSUM(partial, part);
            OutTile combined;
            if constexpr (IsMax) TMAX(combined, out, partial);
            else TADD(combined, out, partial);
            out = combined;
        }
    }
}

template <is_tile_data_v TileOut, is_tile_data_v TileIn>
inline void fa_kchains_tcvt_packed_x2(TileOut &dst, TileIn &src) {
    static_assert(TileOut::Rows == TileIn::Rows,
                  "packed TCVT must preserve rows");
    static_assert(TileIn::Cols == TileOut::Cols * 2,
                  "packed-x2 destination must have half as many columns");
    const size_t validCol = dst.GetValidCol();
    const size_t validRow = dst.GetValidRow();
    asm volatile(
        "BSTART.TEPL 27, %c1\n"
        "B.DATR %c2, RNone\n"
        "B.IOT %3, mask=15, last, ->%0<%Z4>\n"
        "B.DIM %5, 0, ->lb0\n"
        "B.DIM %6, 0, ->lb1\n"
        "B.DIM zero, %c7, ->lb2\n"
        : "=Tr"(dst.data())
        : "i"(type_traits<typename TileIn::DType>::TypeCode),
          "i"(type_traits<typename TileOut::DType>::TypeCode),
          "Tr"(src.data()),
          "i"(tile_type_traits<typename TileOut::TileDType>::TilesizeCode),
          "r"(validCol), "r"(validRow), "i"(TileOut::Cols));
}

// Four-PE cooperative FlashAttention with an internal-accumulator QK chain.
//
// QK splits the head dimension qD into QKChainK-wide pieces:
//
//   Q[M, qD] * K[Skv, qD]^T
//     = sum_kchain Q[M, QKChainK] * K[Tk, QKChainK]^T.
//
// Begin/Middle phases retain the raw FP32 accumulator in the CUBE internal
// accumulator path. Only End publishes the score tile consumed by vector ops.
// PV is intentionally not chained across j: online softmax rescales tO between
// KV blocks, so tO is an architectural vector-visible value at that boundary.
template <typename MatrixDType, typename VectorDType, int PackedFactor,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int QKChainK = 32, int scaleD = qD>
void flash_attention_gmma_kchains_impl(
    VectorDType *outPtr, MatrixDType *qPtr, MatrixDType *kPtr,
    MatrixDType *vPtr) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kStoredQD = qD / PackedFactor;
    constexpr int kStoredSkv = Skv / PackedFactor;
    constexpr int kStoredTk = kTk / PackedFactor;
    constexpr int kStoredChainK = QKChainK / PackedFactor;
    constexpr int kQKChains = qD / QKChainK;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be one or two");
    static_assert(qD % PackedFactor == 0 && kTk % PackedFactor == 0 &&
                      QKChainK % PackedFactor == 0,
                  "logical dimensions must be divisible by PackedFactor");
    static_assert(QKChainK > 0 && qD % QKChainK == 0,
                  "qD must contain an integral number of QK K-chain pieces");
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in 1..128");

    using GmQ = global_tensor<MatrixDType, RowMajor<Sq, kStoredQD>>;
    using GmK = global_tensor<MatrixDType, RowMajor<Skv, kStoredQD>>;
    using GmV = global_tensor<MatrixDType, RowMajor<kStoredSkv, vD>>;
    using GmO = global_tensor<VectorDType, RowMajor<Sq, vD>>;

    // Q is loaded as [M, Kc]. K remains in natural row-major [Tk, Kc]
    // storage. Shared B with TransB=0 consumes this physical [N,K] as K^T.
    using TileQMatrix =
        SharedMatrixLeft<MatrixDType, kTileRows, kStoredChainK,
                         kGroupM, kStoredChainK>;
    using TileKMatrix =
        SharedMatrixRight<MatrixDType, kTk, kStoredChainK>;
    using TileVMatrix =
        SharedMatrixRight<MatrixDType, kStoredTk, vD>;
    using TileQ = SharedTile<TileQMatrix>;
    using TileK = SharedTile<TileKMatrix>;
    using TileV = SharedTile<TileVMatrix>;

    using TileWM16 = CubeAccumulatorM16<float, kPeTm, kTk>;
    using TileWM32 = CubeAccumulatorM32<float, kPeTm, kTk>;
    using TileW = std::conditional_t<(kPeTm <= 16), TileWM16, TileWM32>;
    // Partition the local FP32 score tile, preserving its Acc location and
    // CUBE layout. At Tm=128/Tk=128: [32,128] (16KB) -> 8 x [32,16] (2KB).
    constexpr int kReduceBytes = 2048;
    constexpr int kReduceParts =
        (TileW::LogicalTileBytes + kReduceBytes - 1) / kReduceBytes;
    static_assert(kTk % kReduceParts == 0,
                  "score columns must divide evenly into reduction subviews");
    using TileWSub = std::conditional_t<(kPeTm <= 16),
        CubeAccumulatorM16<float, kPeTm, kTk / kReduceParts>,
        CubeAccumulatorM32<float, kPeTm, kTk / kReduceParts>>;
    static_assert(TileWSub::LogicalTileBytes <= kReduceBytes,
                  "reduction subview exceeds 2KB");
    using TileOM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using TileOM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using TileO = std::conditional_t<(kPeTm <= 16), TileOM16, TileOM32>;

    using TilePShardM16 =
        CubeTileM16<MatrixDType, kPeTm, kStoredTk>;
    using TilePShardM32 =
        CubeTileM32<MatrixDType, kPeTm, kStoredTk>;
    using TilePShard =
        std::conditional_t<(kPeTm <= 16), TilePShardM16, TilePShardM32>;
    using TileOCastM16 =
        CubeAccumulatorM16<VectorDType, kPeTm, vD>;
    using TileOCastM32 =
        CubeAccumulatorM32<VectorDType, kPeTm, vD>;
    using TileOCast =
        std::conditional_t<(kPeTm <= 16), TileOCastM16, TileOCastM32>;
    using TileRow = VecTileM32<VectorDType, 32, 1, kPeTm, 1>;

    using ItQ = global_iterator<GmQ, TileQMatrix>;
    using ItK = global_iterator<GmK, TileKMatrix>;
    using ItV = global_iterator<GmV, TileVMatrix>;
    using ItO = global_iterator<GmO, TileOCast>;
    ItQ gIterQ(qPtr);
    ItK gIterK(kPtr);
    ItV gIterV(vPtr);
    ItO gIterO(outPtr);

    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kQKSharedBytes =
        TileQMatrix::LogicalTileBytes + TileKMatrix::LogicalTileBytes;
    constexpr int kPVSharedBytes = TileVMatrix::LogicalTileBytes;
    static_assert(kQKSharedBytes <= kSharedTRegBytes,
                  "one Q/K K-chain piece exceeds SharedTReg capacity");
    static_assert(kPVSharedBytes <= kSharedTRegBytes,
                  "V tile exceeds SharedTReg capacity");

    const float scale = 1.0f / sqrt(static_cast<float>(scaleD));
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kKVBlocks = (Skv + kTk - 1) / kTk;
    // V is physically [K,N], so Shared B requires TransB=1 for PV.
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int i = 0; i < kQBlocks; ++i) {
        TileRow tMax;
        TileRow tSum;
        TileO tO;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

#pragma clang loop unroll(full)
        for (int j = 0; j < kKVBlocks; ++j) {
            TileW tW;

            // QK K-chain. K is loaded from natural [Skv, qD] row-major GM;
            // each piece is [kTk, QKChainK], the TransB=0 Shared-B layout.
#pragma clang loop unroll(full)
            for (int kc = 0; kc < kQKChains; ++kc) {
                TileQ tQ;
                TileK tK;
                auto gQ = gIterQ(i, kc);
                auto gK = gIterK(j, kc);
                TLOAD<TileQMatrix, 1>(tQ, gQ);
                TLOAD<TileKMatrix, 1>(tK, gK);

                if constexpr (kQKChains == 1) {
                    TMATMUL_INTERNAL_ACC<
                        TMatmulInternalAccPhase::Single>(tW, tQ, tK);
                } else {
                    if (kc == 0) {
                        TMATMUL_INTERNAL_ACC<
                            TMatmulInternalAccPhase::Begin>(tW, tQ, tK);
                    } else if (kc == kQKChains - 1) {
                        TMATMUL_INTERNAL_ACC<
                            TMatmulInternalAccPhase::End>(tW, tQ, tK);
                    } else {
                        TMATMUL_INTERNAL_ACC<
                            TMatmulInternalAccPhase::Middle>(tW, tQ, tK);
                    }
                }
            }

            TMULS(tW, tW, scale);

            TileRow tLocalMax;
            fa_kchains_subview_reduce<true, kReduceParts, TileWSub>(
                tLocalMax, tW);
            TileRow tNewMax;
            TileRow tScale;
            if (j == 0) {
                tNewMax = tLocalMax;
            } else {
                TMAX(tNewMax, tMax, tLocalMax);
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
                TROWEXPANDMUL(tO, tO, tScale);
            }

            TROWEXPANDEXPDIF(tW, tW, tNewMax);
            TileRow tLocalSum;
            // Create fresh views of exp(score - newMax), not pre-exp scores.
            // Partitioned summation may change FP32 rounding versus one sum.
            fa_kchains_subview_reduce<false, kReduceParts, TileWSub>(
                tLocalSum, tW);
            TileRow tNewSum;
            if (j == 0) {
                tNewSum = tLocalSum;
            } else {
                TFMA(tNewSum, tSum, tScale, tLocalSum);
            }

            TileV tV;
            auto gV = gIterV(j, 0);
            TLOAD<TileVMatrix, 1>(tV, gV);
            TilePShard tP;
            if constexpr (PackedFactor == 2) {
                fa_kchains_tcvt_packed_x2(tP, tW);
            } else {
                TCVT(tP, tW);
            }

            // This is not a hidden K-chain boundary: tO was rescaled by the
            // vector engine above and therefore must remain an explicit C.
            if (j == 0) {
                TMATMUL(tO, tP, tV, pvOptions, kGroupM);
            } else {
                TMATMUL_ACC(tO, tO, tP, tV, pvOptions, kGroupM);
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        TROWEXPANDDIV(tO, tO, tSum);
        auto gO = gIterO(i * kPeNum + tid, 0);
        if constexpr (std::is_same_v<VectorDType, float>) {
            TSTORE_CUBE(gO, tO);
        } else {
            TileOCast tOCast;
            TCVT(tOCast, tO);
            TSTORE_CUBE(gO, tOCast);
        }
    }
}

template <typename DType, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int QKChainK = 32, int scaleD = qD>
void flash_attention_gmma_kchains_pto(DType *outPtr, DType *qPtr,
                                      DType *kPtr, DType *vPtr) {
    flash_attention_gmma_kchains_impl<
        DType, DType, 1, Sq, Skv, qD, vD, kTm, kTk, QKChainK, scaleD>(
        outPtr, qPtr, kPtr, vPtr);
}
