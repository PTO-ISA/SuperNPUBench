#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

// Algorithm D: log-domain softmax + MXFP4 quantization.
// Denominator: independent Vector sums per group, rescaled against each
// group's current log reference; align and combine the groups on the final KV block.
// =====================================================================
// Per 32-column MX group:
//
//   G   = groupmax(s)                     (fused B.FPATR GroupMax, final BF16 D)
//   groupLogReference = G - ln4
//   groupExpValues = exp(s - groupLogReference) = 4*exp(s - G)
//                    (FP4 input; group peak 4 avoids E2M1 saturation)
//   groupScaleValues = exp(groupLogReference - R)
//                      (scale before E8M0; R = online row max)
//   E   = E8M0_floor(groupScaleValues)
//   fp4 = E2M1(groupExpValues)
//
// Dequant: fp4 * E = t * floor_pow2(m)/m,  m = exp(G - R),  t = exp(s - R).
//
// Online accumulation: FP32 PV accumulator whose add AND online rescale are
// folded into `TMATMUL_MX_ACC` via the Cube CScale (C *= 2^-CScale).  For that
// the row max is snapped toward the ln2 lattice in BF16. The CScale exponent
// is derived from the rounded BF16 difference; this is approximate attention.
// Scores intentionally omit the usual 1/sqrt(qD) attention factor.
namespace fa_lowp_algD {
using namespace pto;

// GroupMax comes directly from the fused BF16 CUBE CELL output (GM2).

constexpr int kPeNum = 4;
constexpr int kMxGroup = 32;
constexpr int kPackedFactor = 2;
constexpr float kLn4 = 1.3862943611198906f;    // ln(4)
constexpr float kInvLn2 = 1.4426950408889634f;  // 1/ln(2)
constexpr float kLn2 = 0.6931471805599453f;     // ln(2)

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void run(__bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
                       const __fp4_e2m1x2 *kPtr, const __fp4_e2m1x2 *vPtr,
                       const __fp8_e8m0 *qScalePtr, const __fp8_e8m0 *kScalePtr,
                       const __fp8_e8m0 *vScalePtr) {
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeM = 32;
    constexpr int kStoredQD = qD / kPackedFactor;
    constexpr int kStoredTk = kTk / kPackedFactor;
    constexpr int kQScaleCols = qD / kMxGroup;
    constexpr int kPScaleCols = kTk / kMxGroup;
    constexpr int kPaddedQScaleCols = ((kQScaleCols + 31) / 32) * 32;
    constexpr int kPaddedPScaleCols = ((kPScaleCols + 31) / 32) * 32;
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kQBatch = 2;
    static_assert(kQBlocks % kQBatch == 0,
                  "dblQ requires pairs of distinct 128-row Q tiles");
    constexpr int kKVBlocks = Skv / kTk;
    // Continuous lookahead: kPreloadCount is the in-flight distance N (how many
    // KV blocks ahead K/KScale are fetched); one extra slot lets iteration kb
    // issue block kb+N into the slot consumed at kb-1 without a WAR stall.
    constexpr int kPreloadCount = 1;
    constexpr int kPreloadDepth = 2;
    const uint32_t tid = get_thread_idx();

    static_assert(kTm == 128,
                  "AlgoD requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    static_assert(kTk == 256, "AlgoD uses 256-column KV tiles");
    static_assert(qD == 128 && vD == 128, "supported head dimensions are 128");
    static_assert(Sq >= 256 && Skv >= 256, "minimum sequence lengths are 256");
    static_assert((kPScaleCols & (kPScaleCols - 1)) == 0,
                  "group partitioning requires a power-of-two block count");

    using QSlice = global_tensor<__fp4_e2m1x2, RowMajor<kGroupM, kStoredQD>>;
    using KSlice = global_tensor<__fp4_e2m1x2, RowMajor<kTk, kStoredQD>>;
    using VSlice = global_tensor<__fp4_e2m1x2, RowMajor<kTk, vD / 2>>;
    using GmQScale = global_tensor<__fp8_e8m0, RowMajor<Sq, kQScaleCols>>;
    using GmKScale = global_tensor<__fp8_e8m0, RowMajor<Skv, kQScaleCols>>;
    // PTO #343: Shared ScaleB is always physical [N,G_B], independent of
    // TransB, so the host stores V's scale row-major as [vD, Skv/32] = [N,G_B]
    // (the transposed view of the natural per-32-row-group quantization).
    using GmVScale = global_tensor<__fp8_e8m0, RowMajor<vD, Skv / kMxGroup>>;
    using GmO = global_tensor<__bf16, RowMajor<Sq, vD>>;

    using QMatrix = SharedMatrixLeft<__fp4_e2m1x2, 128, qD, 128, qD>;
    using KMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, qD>;
    using VMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, vD>;
    using QTile = SharedTile<QMatrix>;
    using KTile = SharedTile<KMatrix>;
    using VTile = SharedTile<VMatrix>;

    using QScaleMatrix = SharedMatrixLeft<
        __fp8_e8m0, 128, kPaddedQScaleCols, 128, kQScaleCols>;
    using KScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, kTk, kPaddedQScaleCols, kTk, kQScaleCols>;
    using VScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, vD, kPaddedPScaleCols, vD, kPScaleCols>;
    using QScaleTile = SharedTile<QScaleMatrix>;
    using KScaleTile = SharedTile<KScaleMatrix>;
    using VScaleTile = SharedTile<VScaleMatrix>;

    // BF16 score / softmax carriers.
    using Bf16ScoreTile = CubeTileM32<__bf16, kPeM, kTk>;
    using Bf16ScoreGroupTile = CubeTileM32<__bf16, kPeM, kMxGroup>;
    using Bf16RowValueTile = VecTileM32<__bf16, kPeM, 2, kPeM, 1>;
    using Fp32RowValueTile = VecTileM32<float, kPeM, 1, kPeM, 1>;
    using IntRowTile = VecTileM32<int32_t, kPeM, 1, kPeM, 1>;
    // Fused-aux outputs: PTO #346 / TileOP #227 reduce the final encoded D, so
    // RowMax/GroupMax carry the effective D dtype (BF16 here, not AccType).
    // Fused GroupMaxOut is consumed directly as packed BF16 CELLs.
    using Bf16GroupMaxTile = VecTileM32<__bf16, kPeM, kPScaleCols>;
    using WideBf16GroupReductionTile =
        VecTileM32<__bf16, kPeM, kMxGroup, kPeM, 1>;
    // CScale: U8 CUBE_M32 Mx1 raw exponent (one 128 B CELL).  C *= 2^-CScale.
    using CScaleTile = Tile<Location::Vec, uint8_t, kPeM, 4,
                            BLayout::CubeM32, kPeM, 1>;

    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    // P scale: one E8M0 per 32-column group, i.e. Local ScaleA valid
    // [M, ceil(K/32)].  It is ONE CUBE_M32 tile whose G scale bytes are packed
    // contiguously per row (physical Cols = 4*G = the 128 B CELL minimum * G):
    // a TCVT of the whole [M,G] bf16 vector packs all G scales in one shot,
    // whereas TASSEMBLY of per-group 1-valid-column Scaling fragments does not
    // land off slot 0 (the assembled carrier kept only group 0 and zeroed the
    // rest, dropping PV groups 1..G-1).
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
                        kPeM, 4 * kPScaleCols, BLayout::CubeM32,
                        kPeM, kPScaleCols>;

    // FP32 PV accumulator so TMATMUL_MX_ACC folds the add + CScale rescale.
    using Fp32PvTile = CubeAccumulatorM32<float, kPeM, vD>;
    using Bf16StoreTile = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16StoreTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KScaleIter kScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    OIter outIter(outPtr);

    static_assert(P::LogicalTileBytes == PBlock::LogicalTileBytes * kPScaleCols);
    static_assert(PScale::ValidCol == kPScaleCols);

    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int qbBase = 0; qbBase < kQBlocks; qbBase += kQBatch) {
        // Each Q owns its online state, including a distinct L0C PV chain.
        Fp32PvTile weightedValueSums[kQBatch];
        Bf16RowValueTile runningMaxGrids[kQBatch];
        Bf16RowValueTile runningSums[kQBatch];
        Bf16RowValueTile perQueryGroupSums[kQBatch][kPScaleCols];
        Bf16GroupMaxTile previousGroupLogReferences[kQBatch];
        QTile queries[kQBatch];
        QScaleTile queryScales[kQBatch];
#pragma clang loop unroll(full)
        for (int qi = 0; qi < kQBatch; ++qi) {
            TEXPANDS(weightedValueSums[qi], 0.0f);
            const int qb = qbBase + qi;
            QSlice gQ(const_cast<__fp4_e2m1x2 *>(qPtr) + qb * kGroupM * kStoredQD);
            auto gQS = qScaleIter(qb, 0);
            TLOAD<QMatrix, 1>(queries[qi], gQ);
            TLOAD<QScaleMatrix, 1>(queryScales[qi], gQS);
        }

        // One-block K/KScale lookahead; alternating slots preserve the current
        // operands while the next block is loaded.
        KTile kRing[kPreloadDepth];
        KScaleTile kScaleRing[kPreloadDepth];
#pragma clang loop unroll(full)
        for (int preload = 0; preload < kPreloadCount; ++preload) {
            KSlice gKPre(const_cast<__fp4_e2m1x2 *>(kPtr) +
                         preload * kTk * kStoredQD);
            auto gKSPre = kScaleIter(preload, 0);
            TLOAD<KMatrix, 1>(kRing[preload], gKPre);
            TLOAD<KScaleMatrix, 1>(kScaleRing[preload], gKSPre);
        }

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
            KTile &kUse = kRing[kb % kPreloadDepth];
            KScaleTile &kScaleUse = kScaleRing[kb % kPreloadDepth];
            // Lookahead is a compile-time-described constant block for the
            // fully unrolled loop; the last kPreloadCount iterations have none.
            if (kb + kPreloadCount < kKVBlocks) {
                KTile &kNext = kRing[(kb + kPreloadCount) % kPreloadDepth];
                KScaleTile &kScaleNext =
                    kScaleRing[(kb + kPreloadCount) % kPreloadDepth];
                KSlice gKNext(const_cast<__fp4_e2m1x2 *>(kPtr) +
                              (kb + kPreloadCount) * kTk * kStoredQD);
                auto gKSNext = kScaleIter(kb + kPreloadCount, 0);
                TLOAD<KMatrix, 1>(kNext, gKNext);
                TLOAD<KScaleMatrix, 1>(kScaleNext, gKSNext);
            }

            // Load K/V only once per pair of different Q tiles.
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) + kb * kStoredTk * vD);
            GmVScale gVS(vScalePtr + kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
#pragma clang loop unroll(full)
            for (int qi = 0; qi < kQBatch; ++qi) {
                const int qb = qbBase + qi;
                auto &q = queries[qi];
                auto &qScale = queryScales[qi];
                auto &weightedValueSum = weightedValueSums[qi];
                auto &runningMaxGrid = runningMaxGrids[qi];
                auto &runningSum = runningSums[qi];
                auto &groupSums = perQueryGroupSums[qi];
                auto &previousGroupLogReference = previousGroupLogReferences[qi];

                // --- QK: BF16 destination + fused RowMax + fused GroupMax -------
                // NOTE: RowMaxInit (`row_max(in,out)` = max(in, rowmax(D))) would
                // remove the explicit TMAX, but ToolOP currently requires
                // RowMaxIn.ValidRow == core-total group_M while the per-PE
                // RowMaxOut.ValidRow == 32 (template_asm.hpp:5081-5094); the two
                // are inconsistent on the cooperative path.
                Bf16ScoreTile score;
                Bf16RowValueTile tLocalMaxB;   // fused RowMaxOut (final D = BF16)
                Bf16GroupMaxTile tGroupMaxB;   // fused GroupMaxOut (final D = BF16)
                auto qkOptions =
                    fixp::bf16().row_max(tLocalMaxB).group_max<32>(tGroupMaxB);
                TMATMUL_MX<3>(score, q, qScale, kUse, kScaleUse, qkOptions);

                // --- online row max on the ln2 lattice ---------------------------
                CScaleTile cscale;
                Bf16RowValueTile tQ, tRg, tNewMaxB, tDiff;
                IntRowTile tIdx;
                // 2) tQ = R/ln2
                TMULS(tQ, tLocalMaxB, static_cast<__bf16>(kInvLn2));
                TCVT<LINX_RDN>(tIdx, tQ);                 // floor(R / ln2)
                TCVT(tRg, tIdx);
                TMULS(tRg, tRg, static_cast<__bf16>(kLn2));  // R snapped to the lattice
                if (kb == 0) {
                    runningMaxGrid = tRg;
                } else {
                    TMAX(tNewMaxB, runningMaxGrid, tRg);
                    TSUB(tDiff, tNewMaxB, runningMaxGrid);
                    TMULS(tDiff, tDiff, static_cast<__bf16>(kInvLn2));
                    TCVT(tIdx, tDiff);
                    TCVT(cscale, tIdx);                   // U8 raw exponent
                    runningMaxGrid = tNewMaxB;
                }

                // --- per-group MXFP4 quantization (per 32-column group) --------------
                auto blocks =
                    TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(score);
                Bf16GroupMaxTile groupLogReference;
                TSUBS(groupLogReference, tGroupMaxB, static_cast<__bf16>(kLn4));
                Bf16GroupMaxTile groupUpdateScale, groupScaleValues;
                if (kb != 0)
                    TEXPDIF(groupUpdateScale, previousGroupLogReference,
                            groupLogReference);
                TROWEXPANDEXPDIF(groupScaleValues, groupLogReference, runningMaxGrid);
                // Assemble the group fragments into the read-only PV input.
                TileArray<PBlock, 1, kPScaleCols> pFragments;
                // Finish denominator work before opening the assembled P parent.
                Bf16ScoreGroupTile groupExpTiles[kPScaleCols];
                Bf16RowValueTile localSum;
                if (kb == kKVBlocks - 1)
                    TEXPANDS(localSum, static_cast<__bf16>(0.0f));
                // A BF16 CELL contains two groups; byte offset selects the slot.
                // Requires the model's CUBE row-broadcast byte selector support:
                // odd BF16 groups select byte 2 of the packed two-group CELL.
                using Bf16GroupPair = VecTileM32<__bf16, kPeM, 2>;
                auto referencePairs =
                    TPARTVIEW<Bf16GroupPair, 1, kPScaleCols / 2>(groupLogReference);
                auto updatePairs =
                    TPARTVIEW<Bf16GroupPair, 1, kPScaleCols / 2>(groupUpdateScale);
                auto finalPairs =
                    TPARTVIEW<Bf16GroupPair, 1, kPScaleCols / 2>(groupScaleValues);
                auto quantizeGroup = [&]<size_t Group>() __attribute__((always_inline)) {
                    auto scoreView = blocks[0][Group];
                    auto referenceView = referencePairs[0][Group / 2];
                    Bf16ScoreGroupTile groupExpValues;
                    TROWEXPANDEXPDIF<Bf16ScoreGroupTile,
                        Bf16ScoreTile, Bf16ScoreGroupTile,
                        Bf16GroupMaxTile, Bf16GroupPair, 2 * (Group % 2)>(
                            groupExpValues, scoreView, referenceView);
                    groupExpTiles[Group] = groupExpValues;
                    WideBf16GroupReductionTile sumWide;
                    TROWSUM(sumWide, groupExpValues);
                    auto groupExpSum = TREDUCEPREFIXVIEW<Bf16RowValueTile>(sumWide);
                    auto &acc = groupSums[Group];
                    if (kb == 0) {
                        // TFMA accepts the reduction prefix directly; assignment
                        // cannot retain this borrowed wide reduction as row state.
                        TMULS(acc, groupExpSum, static_cast<__bf16>(1.0f));
                    } else {
                        auto scaleView = updatePairs[0][Group / 2];
                        Bf16RowValueTile scaled, next;
                        // Round the group rescale before adding the new sum.
                        TROWEXPANDMUL<Bf16RowValueTile, Bf16RowValueTile,
                            Bf16GroupMaxTile, Bf16GroupPair, 2 * (Group % 2)>(
                                scaled, acc, scaleView);
                        TADD(next, scaled, groupExpSum);
                        acc = next;
                    }
                    if (kb == kKVBlocks - 1) {
                        auto scaleView = finalPairs[0][Group / 2];
                        Bf16RowValueTile scaled, next;
                        TROWEXPANDMUL<Bf16RowValueTile, Bf16RowValueTile,
                            Bf16GroupMaxTile, Bf16GroupPair, 2 * (Group % 2)>(
                                scaled, acc, scaleView);
                        TADD(next, scaled, localSum);
                        localSum = next;
                    }
                };
                [&]<size_t... Groups>(std::index_sequence<Groups...>) __attribute__((always_inline)) {
                    (quantizeGroup.template operator()<Groups>(), ...);
                }(std::make_index_sequence<kPScaleCols>{});
                previousGroupLogReference = groupLogReference;
#pragma clang loop unroll(full)
                for (int block = 0; block < kPScaleCols; ++block)
                    TCVT(pFragments[0][block], groupExpTiles[block]);
                P p = TASSEMBLY<P>(std::move(pFragments));

                // Convert all groupScaleValues to E8M0 scales in one instruction.
                // Keeping the valid scales in a single [M,G] CELL gives the
                // contiguous-valid layout TMATMUL_MX expects (see the PScale note).
                PScale pScale;
                TCVT<LINX_RDN>(pScale, groupScaleValues);

                if (kb == kKVBlocks - 1) runningSum = localSum;


                // --- PV: TMATMUL_MX_ACC folds add + CScale rescale ----------------
                // Keep intermediate PV results in InternalAcc (L0C). keep_acc
                // preserves FP32 precision; raw_acc selects the resident output.
                // CScale rescales the previous accumulator before adding P*V.
                // Only the final block publishes to CellReg for normalization.
                if (kb == 0) {
                    if constexpr (kKVBlocks == 1) {
                        TMATMUL_MX<3>(weightedValueSum, p, pScale, v, vScale,
                                      pvOptions, kGroupM);
                    } else {
                        TMATMUL_MX<3>(weightedValueSum, p, pScale, v, vScale,
                                      pvOptions.raw_acc(), kGroupM);
                    }
                } else if (kb != kKVBlocks - 1) {
                    auto pvOptC =
                        pvOptions.raw_acc().acc_hint().cscale(cscale);
                    TMATMUL_MX_ACC<3>(weightedValueSum, weightedValueSum,
                                      p, pScale, v, vScale, pvOptC, kGroupM);
                } else {
                    auto pvOptC = pvOptions.acc_hint().cscale(cscale);
                    TMATMUL_MX_ACC<3>(weightedValueSum, weightedValueSum,
                                      p, pScale, v, vScale, pvOptC, kGroupM);
                }
            }
        }

#pragma clang loop unroll(full)
        for (int qi = 0; qi < kQBatch; ++qi) {
            const int qb = qbBase + qi;
            auto &weightedValueSum = weightedValueSums[qi];
            auto &runningSum = runningSums[qi];
            // Normalize the Cube PV result by the combined Vector group sums.
            Fp32RowValueTile runningSumF32;
            TCVT(runningSumF32, runningSum);
            TROWEXPANDDIV(weightedValueSum, weightedValueSum, runningSumF32);
            Bf16StoreTile oBf16;
            TCVT(oBf16, weightedValueSum);
            auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
            TSTORE_CUBE(gOut, oBf16);
        }
    }
}
}  // namespace fa_lowp_algD
