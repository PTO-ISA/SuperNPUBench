#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

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

// QK reduces the full qD in one multiply. PV splits each KV block's kTk
// into PVChainK-wide pieces. Each PV chain completes before its result is
// added to the online-softmax output; no chain crosses the j-loop boundary.
template <typename MatrixDType, typename VectorDType, int PackedFactor,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int PVChainK = 32, int scaleD = qD>
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
    constexpr int kStoredChainK = PVChainK / PackedFactor;
    constexpr int kPVChains = kTk / PVChainK;

    static_assert(PackedFactor == 1 || PackedFactor == 2,
                  "PackedFactor must be one or two");
    static_assert(qD % PackedFactor == 0 && kTk % PackedFactor == 0 &&
                      PVChainK % PackedFactor == 0,
                  "logical dimensions must be divisible by PackedFactor");
    static_assert(PVChainK > 0 && kTk % PVChainK == 0,
                  "kTk must contain an integral number of PV K-chain pieces");
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in 1..128");

    using GmQ = global_tensor<MatrixDType, RowMajor<Sq, kStoredQD>>;
    using GmK = global_tensor<MatrixDType, RowMajor<Skv, kStoredQD>>;
    using GmV = global_tensor<MatrixDType, RowMajor<kStoredSkv, vD>>;
    using GmO = global_tensor<VectorDType, RowMajor<Sq, vD>>;

    // Q is [M,qD], K is [Tk,qD]; QK does not split the head dimension.
    // Shared B with TransB=0 declares this physical [N,K] storage.
    using TileQMatrix =
        SharedMatrixLeft<MatrixDType, kTileRows, kStoredQD,
                         kGroupM, kStoredQD>;
    using TileKMatrix =
        SharedMatrixRight<MatrixDType, kTk, kStoredQD>;
    using TileVMatrix =
        SharedMatrixRight<MatrixDType, vD, kStoredChainK>;
    using TileQ = SharedTile<TileQMatrix>;
    using TileK = SharedTile<TileKMatrix>;
    using TileV = SharedTile<TileVMatrix>;

    using TileWM16 = CubeAccumulatorM16<float, kPeTm, kTk>;
    using TileWM32 = CubeAccumulatorM32<float, kPeTm, kTk>;
    using TileW = std::conditional_t<(kPeTm <= 16), TileWM16, TileWM32>;
    using TileOM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using TileOM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using TileO = std::conditional_t<(kPeTm <= 16), TileOM16, TileOM32>;

    using TilePShardM16 =
        CubeTileM16<MatrixDType, kPeTm, kStoredChainK>;
    using TilePShardM32 =
        CubeTileM32<MatrixDType, kPeTm, kStoredChainK>;
    using TilePShard =
        std::conditional_t<(kPeTm <= 16), TilePShardM16, TilePShardM32>;
    // Each probability slice is [peTm,PVChainK], with the parent's Acc layout.
    using TileWPV = std::conditional_t<(kPeTm <= 16),
        CubeAccumulatorM16<float, kPeTm, PVChainK>,
        CubeAccumulatorM32<float, kPeTm, PVChainK>>;
    using TileOCastM16 =
        CubeAccumulatorM16<VectorDType, kPeTm, vD>;
    using TileOCastM32 =
        CubeAccumulatorM32<VectorDType, kPeTm, vD>;
    using TileOCast =
        std::conditional_t<(kPeTm <= 16), TileOCastM16, TileOCastM32>;
    // PTO #311: TROWSUM/TROWMAX require the dst to have the same physical
    // storage as the src (destinationShape = [cellRows, source.col]).
    // TileReduce mirrors TileW's [kPeTm, kTk] shape with ValidCol=1.
    using TileReduceM16 = VecTileM16<float, kPeTm, kTk, kPeTm, 1>;
    using TileReduceM32 = VecTileM32<float, kPeTm, kTk, kPeTm, 1>;
    using TileReduce =
        std::conditional_t<(kPeTm <= 16), TileReduceM16, TileReduceM32>;

    // TROWEXPAND* require a single-column (Cols=1) broadcast source.
    // TileRow is the compact form; TCVT copies from TileReduce after the
    // reduction.
    using TileRowM16 = VecTileM16<float, kPeTm, 1, kPeTm, 1>;
    using TileRowM32 = VecTileM32<float, kPeTm, 1, kPeTm, 1>;
    using TileRow =
        std::conditional_t<(kPeTm <= 16), TileRowM16, TileRowM32>;

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
                  "full Q/K tiles exceed SharedTReg capacity");
    static_assert(kPVSharedBytes <= kSharedTRegBytes,
                  "V tile exceeds SharedTReg capacity");

    const float scale = 1.0f / sqrt(static_cast<float>(scaleD));
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kKVBlocks = (Skv + kTk - 1) / kTk;
    // PTO/TileOP Shared-B storage contract:
    //   TransB=0 declares physical [N,K]; TransB=1 declares physical [K,N].
    // K is stored [N=Tk,K=qD], while V is stored [K=PVChainK,N=vD].
    constexpr auto qkOptions = fixp::keep_acc();
    constexpr auto pvOptions = fixp::keep_acc();

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

            TileQ tQ;
            TileK tK;
            auto gQ = gIterQ(i, 0);
            auto gK = gIterK(j, 0);
            TLOAD<TileQMatrix, 1>(tQ, gQ);
            TLOAD<TileKMatrix, 1>(tK, gK);
            TMATMUL(tW, tQ, tK, qkOptions);

            TMULS(tW, tW, scale);

            TileReduce tLocalMaxR;
            TROWMAX(tLocalMaxR, tW);
            // PTO #311 keeps the one-column reduction result in the prefix
            // CELL of the wide carrier.  The dedicated reduction-prefix view
            // lets TMAX consume that CELL through B.SUBVIEW without the old
            // wide-to-compact TCVT copy.
            auto tLocalMax = TREDUCEPREFIXVIEW<TileRow>(tLocalMaxR);
            TileRow tNewMax;
            TileRow tScale;
            // tMax starts at the softmax sentinel, so the same TMAX is valid
            // for the first block and all subsequent online-max updates.
            TMAX(tNewMax, tMax, tLocalMax);
            if (j != 0) {
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
                TROWEXPANDMUL(tO, tO, tScale);
            }

            TROWEXPANDEXPDIF(tW, tW, tNewMax);
            TileReduce tLocalSumR;
            TROWSUM(tLocalSumR, tW);
            // As with row-max, the row-sum values already occupy the prefix
            // CELL of the wide PTO #311 reduction carrier.  Keep them as a
            // zero-copy view and consume the view with the mixed tile/view
            // TADD overload instead of materializing a compact tile by TCVT.
            auto tLocalSum = TREDUCEPREFIXVIEW<TileRow>(tLocalSumR);
            TileRow tNewSum;
            if (j == 0) {
                // tSum was initialized to zero, so this also materializes the
                // first local sum without a dedicated copy instruction.
                TADD(tNewSum, tSum, tLocalSum);
            } else {
                // TileOP does not yet provide TFMA with a reduction-prefix
                // operand.  Split oldSum*scale + localSum into two operations
                // so the final TADD can consume tLocalSum through B.SUBVIEW.
                // This has separate MUL/ADD rounding instead of fused rounding.
                TileRow tScaledSum;
                TMUL(tScaledSum, tSum, tScale);
                TADD(tNewSum, tScaledSum, tLocalSum);
            }

            // Partition P along the PV reduction dimension; each V load
            // selects the matching rows [j*kTk + kc*PVChainK, ...).
            auto probabilities = TPARTVIEW<TileWPV, 1, kPVChains>(tW);
            TileO tPV;
#pragma clang loop unroll(full)
            for (int kc = 0; kc < kPVChains; ++kc) {
                auto pView = probabilities[0][kc];
                TilePShard tP;
                // CUBE subview TCVT is not supported by the current API.
                // Materialize the nonnegative probability slice before TCVT.
                TileWPV tProbability;
                TMULS(tProbability, pView, 1.0f);
                if constexpr (PackedFactor == 2) {
                    fa_kchains_tcvt_packed_x2(tP, tProbability);
                } else {
                    TCVT(tP, tProbability);
                }
                TileV tV;
                auto gV = gIterV(j * kPVChains + kc, 0);
                TLOAD<TileVMatrix, 1>(tV, gV);

                // Direct CCTRL configuration; current ACC API keeps C=D.
                if constexpr (kPVChains == 1) {
                    TMATMUL(tPV, tP, tV, pvOptions, kGroupM);
                } else if (kc == 0) {
                    TMATMUL(tPV, tP, tV, pvOptions.raw_acc(), kGroupM);
                } else if (kc == kPVChains - 1) {
                    TMATMUL_ACC(tPV, tPV, tP, tV,
                                pvOptions.acc_hint(), kGroupM);
                } else {
                    TMATMUL_ACC(tPV, tPV, tP, tV,
                                pvOptions.raw_acc().acc_hint(), kGroupM);
                }
            }
            // tO already holds alpha * previous_O for j>0.
            if (j == 0) tO = tPV;
            else TADD(tO, tO, tPV);

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
          int PVChainK = 32, int scaleD = qD>
void flash_attention_gmma_kchains_pto(DType *outPtr, DType *qPtr,
                                      DType *kPtr, DType *vPtr) {
    flash_attention_gmma_kchains_impl<
        DType, DType, 1, Sq, Skv, qD, vD, kTm, kTk, PVChainK, scaleD>(
        outPtr, qPtr, kPtr, vPtr);
}
