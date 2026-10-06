#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

// Final Algorithm-C-only log-domain softmax + MXFP4 quantization path.
//
// Per 32-column MX group:
//   G   = fused GroupMaxOut(s)             (BF16)
//   P3  = G - ln4
//   P4  = exp(s - P3) = 4*exp(s - G)      (E2M1 payload)
//   P5  = exp(P3 - R)                     (E8M0 scale source)
//
// The denominator is evaluated by Cube as P x 1.  GroupMaxOut is consumed
// directly in BF16: a SubTileView selects a 128-byte CUBE_M32 CELL and the
// TROWEXPANDEXPDIF BroadcastByteOffset selects one of its two BF16 row slots.
namespace fa_lowp_algC_final {
using namespace pto;

constexpr int kPeNum = 4;
constexpr int kMxGroup = 32;
constexpr int kPackedFactor = 2;
constexpr float kLn4 = 1.3862943611198906f;
constexpr float kInvLn2 = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void fa_lowp_algC_final_impl(
    __bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
    const __fp4_e2m1x2 *kPtr, const __fp4_e2m1x2 *vPtr,
    const __fp8_e8m0 *qScalePtr, const __fp8_e8m0 *kScalePtr,
    const __fp8_e8m0 *vScalePtr) {
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeM = 32;
    constexpr int kStoredTk = kTk / kPackedFactor;
    constexpr int kQScaleCols = qD / kMxGroup;
    constexpr int kPScaleCols = kTk / kMxGroup;
    constexpr int kGroupMaxCellCount = (kPScaleCols + 1) / 2;
    constexpr int kPaddedQScaleCols = ((kQScaleCols + 31) / 32) * 32;
    constexpr int kPaddedPScaleCols = ((kPScaleCols + 31) / 32) * 32;
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kKVBlocks = Skv / kTk;
    const uint32_t tid = get_thread_idx();

    static_assert(kGroupM == 128,
                  "fa_lowp_algC_final requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    static_assert((kPScaleCols & (kPScaleCols - 1)) == 0,
                  "group partitioning requires a power-of-two block count");

    using QSlice =
        global_tensor<__fp4_e2m1x2,
                      RowMajor<kGroupM, qD / kPackedFactor>>;
    using KSlice =
        global_tensor<__fp4_e2m1x2, RowMajor<kTk, qD / kPackedFactor>>;
    using VSlice =
        global_tensor<__fp4_e2m1x2, RowMajor<kTk, vD / kPackedFactor>>;
    using GmQScale = global_tensor<__fp8_e8m0, RowMajor<Sq, kQScaleCols>>;
    using GmKScale = global_tensor<__fp8_e8m0, RowMajor<Skv, kQScaleCols>>;
    using GmVScale =
        global_tensor<__fp8_e8m0, RowMajor<vD, Skv / kMxGroup>>;
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

    using Bf16ScoreTile = CubeTileM32<__bf16, kPeM, kTk>;
    using Bf16ScoreGroupTile = CubeTileM32<__bf16, kPeM, kMxGroup>;
    using Bf16RowValueTile = VecTileM32<__bf16, kPeM, 2, kPeM, 1>;
    using Fp32RowValueTile = VecTileM32<float, kPeM, 1, kPeM, 1>;
    using IntRowTile = VecTileM32<int32_t, kPeM, 1, kPeM, 1>;

    // Fused GroupMaxOut is [M,G] BF16.  A CUBE_M32 row slice is four bytes,
    // hence each 128-byte CELL contains two BF16 group values per row.
    using Bf16GroupMaxTile = VecTileM32<__bf16, kPeM, kPScaleCols>;
    using Bf16GroupMaxCellTile =
        VecTileM32<__bf16, kPeM, 2, kPeM, 1>;
    using Bf16GroupMaxCellView =
        region::SubTileView<Bf16GroupMaxTile, Bf16GroupMaxCellTile>;

    using CScaleTile = Tile<Location::Vec, uint8_t, kPeM, 4,
                            BLayout::CubeM32, kPeM, 1>;
    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
                        kPeM, 4 * kPScaleCols, BLayout::CubeM32,
                        kPeM, kPScaleCols>;

    using Fp32PvTile = CubeAccumulatorM32<float, kPeM, vD>;
    using Bf16StoreTile = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using VPrimeMatrix = SharedMatrixRight<__bf16, kTk, 1>;
    using VPrimeTile = SharedTile<VPrimeMatrix>;
    using VPrimeGm = global_tensor<__bf16, RowMajor<kTk, 1>>;
    using Fp32DenomTile = CubeAccumulatorM32<float, kPeM, 1>;

    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16StoreTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KScaleIter kScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    OIter outIter(outPtr);

    static_assert(P::LogicalTileBytes ==
                  PBlock::LogicalTileBytes * kPScaleCols);
    static_assert(PScale::ValidCol == kPScaleCols);
    static_assert(Bf16GroupMaxTile::LogicalTileBytes ==
                  Bf16GroupMaxCellTile::LogicalTileBytes *
                      kGroupMaxCellCount);

    constexpr auto pvOptions = fixp::keep_acc().transpose_b();
    constexpr auto denomOptions = fixp::keep_acc().transpose_b();

    // V' is a constant all-one Shared right operand, so P x V' is the row sum
    // of the dequantized P tile and remains on the Cube path.
    static __bf16 kVPrimeOnes[kTk];
#pragma clang loop unroll(full)
    for (int i = 0; i < kTk; ++i)
        kVPrimeOnes[i] = static_cast<__bf16>(1.0f);
    VPrimeTile vPrime;
    VPrimeGm gVPrime(kVPrimeOnes);
    TLOAD<VPrimeMatrix, 1>(vPrime, gVPrime);

#pragma clang loop unroll(full)
    for (int qb = 0; qb < kQBlocks; ++qb) {
        Fp32PvTile weightedValueSum;
        Fp32DenomTile weightedDenom;
        Bf16RowValueTile runningMaxGrid;
        Bf16RowValueTile oldScale;
        TEXPANDS(weightedValueSum, 0.0f);
        TEXPANDS(weightedDenom, 0.0f);
        TEXPANDS(runningMaxGrid, static_cast<__bf16>(-1.0e30f));

        QTile q;
        QScaleTile qScale;
        QSlice gQ(const_cast<__fp4_e2m1x2 *>(qPtr) +
                  qb * kGroupM * (qD / kPackedFactor));
        auto gQS = qScaleIter(qb, 0);
        TLOAD<QMatrix, 1>(q, gQ);
        TLOAD<QScaleMatrix, 1>(qScale, gQS);

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
            KTile k;
            KScaleTile kScale;
            KSlice gK(const_cast<__fp4_e2m1x2 *>(kPtr) +
                      kb * kTk * (qD / kPackedFactor));
            auto gKS = kScaleIter(kb, 0);
            TLOAD<KMatrix, 1>(k, gK);
            TLOAD<KScaleMatrix, 1>(kScale, gKS);

            Bf16ScoreTile score;
            Bf16RowValueTile tLocalMaxB;
            Bf16GroupMaxTile tGroupMaxB;
            auto qkOptions =
                fixp::bf16().row_max(tLocalMaxB).group_max<32>(tGroupMaxB);
            TMATMUL_MX<3>(score, q, qScale, k, kScale, qkOptions);

            CScaleTile cscale;
            Bf16RowValueTile tQ, tRg, tNewMaxB, tDiff;
            IntRowTile tIdx;
            TMULS(tQ, tLocalMaxB, static_cast<__bf16>(kInvLn2));
            TCVT<LINX_RDN>(tIdx, tQ);
            TCVT(tRg, tIdx);
            TMULS(tRg, tRg, static_cast<__bf16>(kLn2));
            if (kb == 0) {
                runningMaxGrid = tRg;
            } else {
                TMAX(tNewMaxB, runningMaxGrid, tRg);
                TROWEXPANDEXPDIF(oldScale, runningMaxGrid, tNewMaxB);
                TSUB(tDiff, tNewMaxB, runningMaxGrid);
                TMULS(tDiff, tDiff, static_cast<__bf16>(kInvLn2));
                TCVT(cscale, tDiff);
                runningMaxGrid = tNewMaxB;
            }

            auto scoreGroups =
                TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(score);
            // Compute every group's P3 = G-ln4 once.  The group loop consumes
            // one BF16 slot directly from this tile; the P-scale path below
            // reuses the complete [M,G] result.
            Bf16GroupMaxTile p3Group;
            TSUBS(p3Group, tGroupMaxB, static_cast<__bf16>(kLn4));
            Bf16RowValueTile zero;
            TEXPANDS(zero, static_cast<__bf16>(0.0f));
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            // Each BF16 GroupMax CELL contains two logical group values.  Loop
            // by CELL so the two row-broadcast byte offsets remain compile-time
            // constants (0 and sizeof(BF16)=2), as required by PR #247.
#pragma clang loop unroll(full)
            for (int cell = 0; cell < kPScaleCols / 2; ++cell) {
                const int evenGroup = 2 * cell;
                const int oddGroup = evenGroup + 1;
                Bf16GroupMaxCellView p3Cell(
                    p3Group, 0, cell, kGroupMaxCellCount);

                Bf16RowValueTile evenP3;
                TROWEXPANDADD<
                    Bf16RowValueTile,
                    Bf16RowValueTile,
                    Bf16GroupMaxTile,
                    Bf16GroupMaxCellTile,
                    0>(evenP3, zero, p3Cell);
                Bf16ScoreGroupTile evenP4;
                auto evenScore = scoreGroups[0][evenGroup];
                TROWEXPANDEXPDIF(evenP4, evenScore, evenP3);
                TCVT(pFragments[0][evenGroup], evenP4);

                Bf16RowValueTile oddP3;
                TROWEXPANDADD<
                    Bf16RowValueTile,
                    Bf16RowValueTile,
                    Bf16GroupMaxTile,
                    Bf16GroupMaxCellTile,
                    sizeof(__bf16)>(oddP3, zero, p3Cell);
                Bf16ScoreGroupTile oddP4;
                auto oddScore = scoreGroups[0][oddGroup];
                TROWEXPANDEXPDIF(oddP4, oddScore, oddP3);
                TCVT(pFragments[0][oddGroup], oddP4);
            }

            // A power-of-two group count can only be odd when it is one.
            // Keep that legal shape working without generating an odd slot.
            if constexpr ((kPScaleCols % 2) != 0) {
                constexpr int kLastGroup = kPScaleCols - 1;
                constexpr int kLastCell = kLastGroup / 2;
                Bf16GroupMaxCellView p3Cell(
                    p3Group, 0, kLastCell, kGroupMaxCellCount);
                Bf16RowValueTile p3;
                TROWEXPANDADD<
                    Bf16RowValueTile,
                    Bf16RowValueTile,
                    Bf16GroupMaxTile,
                    Bf16GroupMaxCellTile,
                    0>(p3, zero, p3Cell);
                Bf16ScoreGroupTile p4;
                auto scoreView = scoreGroups[0][kLastGroup];
                TROWEXPANDEXPDIF(p4, scoreView, p3);
                TCVT(pFragments[0][kLastGroup], p4);
            }

            P p = TASSEMBLY<P>(std::move(pFragments));

            // Build all E8M0 group scales together so TMATMUL_MX sees a single
            // contiguous [M,G] Scaling tile.
            Bf16GroupMaxTile p5Group;
            PScale pScale;
            TROWEXPANDEXPDIF(p5Group, p3Group, runningMaxGrid);
            TCVT<LINX_RDN>(pScale, p5Group);

            pto_matmul_detail::NoScaleOperand noVPrimeScale;
            if (kb == 0) {
                TMATMUL_MX<1>(weightedDenom, p, pScale, vPrime,
                              noVPrimeScale, denomOptions, kGroupM);
            } else {
                Fp32RowValueTile oldScaleF32;
                TCVT(oldScaleF32, oldScale);
                TROWEXPANDMUL(weightedDenom, weightedDenom, oldScaleF32);
                TMATMUL_MX_ACC<1>(weightedDenom, weightedDenom,
                                  p, pScale, vPrime, noVPrimeScale,
                                  denomOptions, kGroupM);
            }

            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) +
                      kb * kStoredTk * vD);
            GmVScale gVS(vScalePtr + kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
            if (kb == 0) {
                TMATMUL_MX<3>(weightedValueSum, p, pScale, v, vScale,
                              pvOptions, kGroupM);
            } else {
                auto pvOptC =
                    fixp::keep_acc().transpose_b().acc_hint().cscale(cscale);
                TMATMUL_MX_ACC<3>(weightedValueSum, weightedValueSum,
                                  p, pScale, v, vScale, pvOptC, kGroupM);
            }
        }

        TROWEXPANDDIV(weightedValueSum, weightedValueSum, weightedDenom);
        Bf16StoreTile oBf16;
        TCVT(oBf16, weightedValueSum);
        auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
        TSTORE_CUBE(gOut, oBf16);
    }
}

}  // namespace fa_lowp_algC_final
