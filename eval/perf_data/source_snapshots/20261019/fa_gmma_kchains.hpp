#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>

using namespace pto;

// QK reduces the full qD. PV uses the full Tk reduction dimension and carries
// tO across KV blocks with CScale + InternalAcc hints (explicit C is still
// required by the API).
// CScale represents 2^(-u8). The running row-max uses the original score
// value; the denominator uses the same quantized old-state scale as CUBE.
// Logical shape flow per PE (m = kPeTm, one PE owns m query rows):
//   Q[m,qD] x K[Tk,qD]^T -> score[m,Tk] -> P[m,Tk]
//   P[m,Tk] x V[Tk,vD] -> numerator[m,vD]; row max/sum are [m,1].
// Shared Q holds kGroupM rows for the four cooperating PEs; the score,
// online-softmax state, and output accumulator below are PE-local tiles.
template <typename MatrixDType, typename VectorDType,
          int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD, bool AddressPreload = false,
          bool KTilePreload = false>
void flash_attention_gmma_kchains_impl(
    VectorDType *outPtr, MatrixDType *qPtr, MatrixDType *kPtr,
    MatrixDType *vPtr) {
    const uint32_t tid = get_thread_idx();
    constexpr int kPeNum = 4;
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeTm = kGroupM <= 64 ? 16 : 32;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    static_assert(kTm % kGroupM == 0 && Sq % kGroupM == 0,
                  "Tm and Sq must be divisible by cooperative group_M");
    static_assert(kGroupM >= 1 && kGroupM <= 128,
                  "cooperative group_M must be in 1..128");
    static_assert(kPeTm == 32,
                  "CScale requires CUBE_M32; use Tm >= 128 for this kernel");
    static_assert(Skv % kTk == 0,
                  "partial KV blocks require masking before softmax");

    using GmQ = global_tensor<MatrixDType, RowMajor<Sq, qD>>;
    using GmK = global_tensor<MatrixDType, RowMajor<Skv, qD>>;
    using GmV = global_tensor<MatrixDType, RowMajor<Skv, vD>>;
    using GmO = global_tensor<VectorDType, RowMajor<Sq, vD>>;

    // Q is [M,qD], K is [Tk,qD]; QK does not split the head dimension.
    // Shared B with TransB=0 declares this physical [N,K] storage.
    using TileQMatrix =
        SharedMatrixLeft<MatrixDType, kTileRows, qD, kGroupM, qD>;
    using TileKMatrix =
        SharedMatrixRight<MatrixDType, kTk, qD>;
    using TileVMatrix =
        SharedMatrixRight<MatrixDType, kTk, vD>;
    using TileQ = SharedTile<TileQMatrix>;
    using TileK = SharedTile<TileKMatrix>;
    using TileV = SharedTile<TileVMatrix>;

    // QK score/P: logical [m,Tk] in vector precision. PV accumulates in
    // FP32 [m,vD], even when the eventual output is BF16.
    using TileWM16 = CubeTileM16<VectorDType, kPeTm, kTk>;
    using TileWM32 = CubeTileM32<VectorDType, kPeTm, kTk>;
    using TileW = std::conditional_t<(kPeTm <= 16), TileWM16, TileWM32>;
    using TileOM16 = CubeAccumulatorM16<float, kPeTm, vD>;
    using TileOM32 = CubeAccumulatorM32<float, kPeTm, vD>;
    using TileO = std::conditional_t<(kPeTm <= 16), TileOM16, TileOM32>;

    using TilePShardM16 =
        CubeTileM16<MatrixDType, kPeTm, kTk>;
    using TilePShardM32 =
        CubeTileM32<MatrixDType, kPeTm, kTk>;
    using TilePShard =
        std::conditional_t<(kPeTm <= 16), TilePShardM16, TilePShardM32>;
    using TileOCastM16 =
        CubeAccumulatorM16<VectorDType, kPeTm, vD>;
    using TileOCastM32 =
        CubeAccumulatorM32<VectorDType, kPeTm, vD>;
    using TileOCast =
        std::conditional_t<(kPeTm <= 16), TileOCastM16, TileOCastM32>;
    // PTO #311 row-sum carrier: physical [m,Tk], valid [m,1]. Its prefix
    // view exposes the row values without copying them to a compact tile.
    using TileReduceM16 = VecTileM16<VectorDType, kPeTm, kTk, kPeTm, 1>;
    using TileReduceM32 = VecTileM32<VectorDType, kPeTm, kTk, kPeTm, 1>;
    using TileReduce =
        std::conditional_t<(kPeTm <= 16), TileReduceM16, TileReduceM32>;

    // Compact [m,1] row state broadcasts across score[m,Tk] or output[m,vD].
    using TileRowM16 = VecTileM16<VectorDType, kPeTm, 1, kPeTm, 1>;
    using TileRowM32 = VecTileM32<VectorDType, kPeTm, 1, kPeTm, 1>;
    using TileRow =
        std::conditional_t<(kPeTm <= 16), TileRowM16, TileRowM32>;
    // PTO #346 computes fixpipe reductions from final encoded D, so the
    // auxiliary row-max dtype must match the configured vector precision.
    // QK always produces row-max through fixpipe, without a standalone
    // TROWMAX pass over the score tile.
    using TileFixpipeRowMax =
        VecTileM32<VectorDType, 32, 1, kPeTm, 1>;
    using TileRowFp32M16 = VecTileM16<float, kPeTm, 1, kPeTm, 1>;
    using TileRowFp32M32 = VecTileM32<float, kPeTm, 1, kPeTm, 1>;
    using TileRowFp32 =
        std::conditional_t<(kPeTm <= 16), TileRowFp32M16, TileRowFp32M32>;
    using TileIndex = VecTileM32<int32_t, kPeTm, 1, kPeTm, 1>;
    using TileCScale = VecTileM32<uint8_t, kPeTm, 1>;

    using ItQ = global_iterator<GmQ, TileQMatrix>;
    using ItK = global_iterator<GmK, TileKMatrix>;
    using ItV = global_iterator<GmV, TileVMatrix>;
    using ItO = global_iterator<GmO, TileOCast>;
    using KSlice = global_tensor<MatrixDType, RowMajor<kTk, qD>>;
    using VSlice = global_tensor<MatrixDType, RowMajor<kTk, vD>>;
    ItQ gIterQ(qPtr);
    ItK gIterK(kPtr);
    ItV gIterV(vPtr);
    ItO gIterO(outPtr);

    constexpr int kSharedTRegBytes = 256 * 1024;
    constexpr int kQKSharedBytes =
        TileQMatrix::LogicalTileBytes + TileKMatrix::LogicalTileBytes;
    constexpr int kPVSharedBytes = TileVMatrix::LogicalTileBytes;
    // The next K load is issued after the current PV has entered CUBE.
    // Conservatively keep Q, both K slots, and the current V live until
    // that PV completes.
    constexpr int kPreloadSharedBytes =
        TileQMatrix::LogicalTileBytes +
        2 * TileKMatrix::LogicalTileBytes + TileVMatrix::LogicalTileBytes;
    static_assert(kQKSharedBytes <= kSharedTRegBytes,
                  "full Q/K tiles exceed SharedTReg capacity");
    static_assert(kPVSharedBytes <= kSharedTRegBytes,
                  "V tile exceeds SharedTReg capacity");
    static_assert(!KTilePreload || kPreloadSharedBytes <= kSharedTRegBytes,
                  "K lookahead exceeds SharedTReg capacity");

    const float scale = 1.0f / sqrt(static_cast<float>(scaleD));
    constexpr float kInvLn2 = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kKVBlocks = (Skv + kTk - 1) / kTk;
    // PTO/TileOP Shared-B storage contract:
    //   TransB=0 declares physical [N,K]; TransB=1 declares physical [K,N].
    // K is stored [N=Tk,K=qD], while V is stored [K=Tk,N=vD].
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int i = 0; i < kQBlocks; ++i) {
        // Step 1: initialize the online state for these kGroupM Q rows.
        // Each PE owns max/sum[m,1] and an FP32 numerator[m,vD].
        TileRow tMax;
        TileRow tSum;
        TileO tO;
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        // Step 2: load Q[kGroupM,qD] once; all KV blocks reuse it.
        TileQ tQ;
        auto gQ = gIterQ(i, 0);
        TLOAD<TileQMatrix, 1>(tQ, gQ);

        // Address lookahead keeps K/V GM bases in two cursors. With
        // KTilePreload, ping-pong SharedTile slots also stage K[j+1].
        MatrixDType *kCursor = kPtr;
        MatrixDType *vCursor = vPtr;
        TileK kLookahead[2];
        if constexpr (KTilePreload) {
            KSlice gKFirst(kPtr);
            TLOAD<TileKMatrix, 1>(kLookahead[0], gKFirst);
        }

#pragma clang loop unroll(full)
        for (int j = 0; j < kKVBlocks; ++j) {
            // Step 3: consume this block's K[Tk,qD]. The ordinary path loads
            // it here; KTilePreload instead uses the staged ping-pong slot.
            MatrixDType *currentKBase = nullptr;
            MatrixDType *currentVBase = nullptr;
            if constexpr (AddressPreload) {
                currentKBase = kCursor;
                currentVBase = vCursor;
                if (j + 1 < kKVBlocks) {
                    kCursor += kTk * qD;
                    vCursor += kTk * vD;
                    // Retain the two recurrences through the unrolled body.
                    asm volatile("" : "+r"(kCursor), "+r"(vCursor));
                }
            }
            TileW tW;

            TileK tKLocal;
            TileK &tK = KTilePreload ? kLookahead[j & 1] : tKLocal;
            if constexpr (!KTilePreload) {
                if constexpr (AddressPreload) {
                    KSlice gK(currentKBase);
                    TLOAD<TileKMatrix, 1>(tK, gK);
                } else {
                    auto gK = gIterK(j, 0);
                    TLOAD<TileKMatrix, 1>(tK, gK);
                }
            }
            // Step 4: QK produces score[m,Tk] plus fixpipe row max[m,1].
            // Apply 1/sqrt(scaleD) to both outputs before the online update.
            TileFixpipeRowMax tLocalMaxFixpipe;
            TileRow tLocalMax;
            if constexpr (std::is_same_v<VectorDType, float>) {
                auto qkRowMaxOptions =
                    fixp::keep_acc().row_max(tLocalMaxFixpipe);
                TMATMUL(tW, tQ, tK, qkRowMaxOptions);
            } else {
                static_assert(std::is_same_v<VectorDType, __bf16>,
                              "fixpipe QK row-max supports FP32/BF16 vector precision");
                auto qkRowMaxOptions =
                    fixp::bf16().row_max(tLocalMaxFixpipe);
                TMATMUL(tW, tQ, tK, qkRowMaxOptions);
            }
            TMULS(tW, tW, scale);
            TMULS(tLocalMaxFixpipe, tLocalMaxFixpipe, scale);
            if constexpr (std::is_same_v<TileRow, TileFixpipeRowMax>) {
                tLocalMax = tLocalMaxFixpipe;
            } else {
                TCVT(tLocalMax, tLocalMaxFixpipe);
            }
            // Step 5: merge the raw local row maximum into the online max.
            // For later blocks, quantize the max difference to CScale and
            // reconstruct the same old-state multiplier for the denominator.
            TileRow tNewMax, tOldScale;
            TileCScale tCScale;
            TMAX(tNewMax, tMax, tLocalMax);
            if (j != 0) {
                TileRow tDiff;
                TSUB(tDiff, tNewMax, tMax);
                TMULS(tDiff, tDiff, kInvLn2);
                TileIndex tIdx;
                TCVT<LINX_RDN>(tIdx, tDiff);
                TCVT(tCScale, tIdx);
                TCVT(tDiff, tCScale);
                TMULS(tDiff, tDiff, -kLn2);
                TEXP(tOldScale, tDiff);
            }

            // Step 6: broadcast newMax[m,1] over score[m,Tk] to form
            // P[m,Tk] = exp(score-newMax), then reduce P by row.
            TROWEXPANDEXPDIF(tW, tW, tNewMax);
            TileReduce tLocalSumR;
            TROWSUM(tLocalSumR, tW);
            // As with row-max, the row-sum values already occupy the prefix
            // CELL of the wide PTO #311 reduction carrier.  Keep them as a
            // zero-copy view and consume the view with the mixed tile/view
            // TADD overload instead of materializing a compact tile by TCVT.
            auto tLocalSum = TREDUCEPREFIXVIEW<TileRow>(tLocalSumR);
            TileRow tNewSum;
            // Step 7: oldSum[m,1] * 2^(-CScale) + blockSum[m,1].
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
                TMUL(tScaledSum, tSum, tOldScale);
                TADD(tNewSum, tScaledSum, tLocalSum);
            }

            // Step 8: load V[Tk,vD]; convert P[m,Tk] to matrix precision
            // only if CUBE cannot consume the vector-precision P directly.
            TileV tV;
            if constexpr (AddressPreload) {
                VSlice gV(currentVBase);
                TLOAD<TileVMatrix, 1>(tV, gV);
            } else {
                auto gV = gIterV(j, 0);
                TLOAD<TileVMatrix, 1>(tV, gV);
            }

            // P and V must use the configured matrix-input precision. FP32
            // GMMA can consume tW directly; lower-precision modes convert the
            // vector-precision probability tile before entering CUBE.
            const bool last = j == kKVBlocks - 1;
            if constexpr (std::is_same_v<MatrixDType, VectorDType>) {
                if (j == 0) {
                    if (last) TMATMUL(tO, tW, tV, pvOptions, kGroupM);
                    else TMATMUL(tO, tW, tV, pvOptions.raw_acc(), kGroupM);
                } else {
                    auto options = pvOptions.acc_hint().cscale(tCScale);
                    if (last) TMATMUL_ACC(tO, tO, tW, tV, options, kGroupM);
                    else TMATMUL_ACC(
                        tO, tO, tW, tV,
                        pvOptions.raw_acc().acc_hint().cscale(tCScale),
                        kGroupM);
                }
            } else {
                TilePShard tP;
                TCVT(tP, tW);
                if (j == 0) {
                    if (last) TMATMUL(tO, tP, tV, pvOptions, kGroupM);
                    else TMATMUL(tO, tP, tV, pvOptions.raw_acc(), kGroupM);
                } else {
                    auto options = pvOptions.acc_hint().cscale(tCScale);
                    if (last) TMATMUL_ACC(tO, tO, tP, tV, options, kGroupM);
                    else TMATMUL_ACC(
                        tO, tO, tP, tV,
                        pvOptions.raw_acc().acc_hint().cscale(tCScale),
                        kGroupM);
                }
            }

            // Issue PV first: on a full SharedTReg pool, putting K[j+1]
            // ahead of PV[j] makes the lookahead TLOAD block the current
            // calculation. The later TLOAD can still execute concurrently
            // with an already-issued PV on the independent TLSU/CUBE paths.
            if constexpr (KTilePreload) {
                if (j + 1 < kKVBlocks) {
                    KSlice gKNext(kPtr + (j + 1) * kTk * qD);
                    TLOAD<TileKMatrix, 1>(kLookahead[(j + 1) & 1], gKNext);
                }
            }

            // Commit the row state after both P-row-sum and PV have used it.
            tMax = tNewMax;
            tSum = tNewSum;
        }

        // Step 9: normalize numerator[m,vD] by denominator[m,1], convert
        // to the output dtype if needed, then store this PE's m rows.
        if constexpr (std::is_same_v<VectorDType, float>) {
            TROWEXPANDDIV(tO, tO, tSum);
        } else {
            TileRowFp32 tSumFp32;
            TCVT(tSumFp32, tSum);
            TROWEXPANDDIV(tO, tO, tSumFp32);
        }
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
          int scaleD = qD>
void flash_attention_gmma_kchains_pto(DType *outPtr, DType *qPtr,
                                      DType *kPtr, DType *vPtr) {
    flash_attention_gmma_kchains_impl<
        DType, DType, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        outPtr, qPtr, kPtr, vPtr);
}
