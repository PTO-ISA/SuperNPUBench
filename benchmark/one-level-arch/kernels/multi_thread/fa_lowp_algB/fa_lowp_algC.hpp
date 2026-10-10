#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

// Algorithm C: log-domain softmax + MXFP4 quantization.
// Denominator: Cube P x V' with an ordinary TileReg accumulator.
// =====================================================================
// Per 32-column MX group (no algorithm-A / no debug branches):
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
// the row max is kept on the ln2 lattice so the rescale factor is an exact
// power of two.
namespace fa_lowp_logdomain {
using namespace pto;

// GroupMax comes directly from the fused BF16 CUBE CELL output (GM2).

constexpr int kPeNum = 4;
constexpr int kMxGroup = 32;
constexpr int kPackedFactor = 2;
constexpr float kLn4 = 1.3862943611198906f;    // ln(4)
constexpr float kInvLn2 = 1.4426950408889634f;  // 1/ln(2)
constexpr float kLn2 = 0.6931471805599453f;     // ln(2)

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void fa_lowp_algC_impl(__bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
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
    constexpr int kKVBlocks = Skv / kTk;
    const uint32_t tid = get_thread_idx();

    static_assert(kGroupM == 128,
                  "fa_lowp_algB requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    // Tk=256 currently trips the model's raw tile spill carrier check.
    static_assert(kTk == 128,
                  "direct GroupMax mode is currently validated only for Tk=128");
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
    // V' denominator stays in a Cube tile across kb.  This is an ordinary
    // TileReg accumulation (no acc_hint/cscale and therefore no internal-ACC
    // K-chain); online rescaling is applied directly to the Cube tile.
    // Two logical N columns share one FP4 byte; both encode exact one.
    using VPrimeMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, 2>;
    using VPrimeScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, 2, kPaddedPScaleCols, 2, kPScaleCols>;
    using VPrimeScaleTile = SharedTile<VPrimeScaleMatrix>;
    using VPrimeScaleGm = global_tensor<__fp8_e8m0, RowMajor<2, kPScaleCols>>;
    using VPrimeTile = SharedTile<VPrimeMatrix>;
    using VPrimeGm = global_tensor<__fp4_e2m1x2, RowMajor<kTk, 1>>;
    using Fp32DenomTile = CubeAccumulatorM32<float, kPeM, 2>;
    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16StoreTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KScaleIter kScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    OIter outIter(outPtr);

    static_assert(P::LogicalTileBytes == PBlock::LogicalTileBytes * kPScaleCols);
    static_assert(PScale::ValidCol == kPScaleCols);

    constexpr auto pvOptions = fixp::keep_acc().transpose_b();
    constexpr auto denomOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int qb = 0; qb < kQBlocks; ++qb) {
        Fp32PvTile weightedValueSum;
        Bf16RowValueTile runningMaxGrid;  // ln2-lattice row max (BF16 softmax)
        TEXPANDS(weightedValueSum, 0.0f);
        TEXPANDS(runningMaxGrid, static_cast<__bf16>(-1.0e30f));
        Bf16RowValueTile runningSum;  // cross-kb V' denominator in Vector
        Bf16RowValueTile oldScale;    // exact-power-of-two online rescale
        TEXPANDS(runningSum, static_cast<__bf16>(0.0f));
        Fp32DenomTile weightedDenom;
        TEXPANDS(weightedDenom, 0.0f);
        // V' is GM ND -> SharedTReg ND.  It is read by each non-ACC P×V'
        // instruction, but never retained in internalACC.
        static __fp4_e2m1x2 kVPrimeOnes[kTk];
        static __fp8_e8m0 kVPrimeScales[2 * kPScaleCols];
#pragma clang loop unroll(full)
        for (int i = 0; i < kTk; ++i)
            kVPrimeOnes[i].data = 0x22; // two E2M1 ones
#pragma clang loop unroll(full)
        for (int i = 0; i < 2 * kPScaleCols; ++i)
            kVPrimeScales[i].data = 0x7f; // E8M0 one
        VPrimeTile vPrime;
        VPrimeGm gVPrime(kVPrimeOnes);
        TLOAD<VPrimeMatrix, 1>(vPrime, gVPrime);
        VPrimeScaleTile vPrimeScale;
        VPrimeScaleGm gVPrimeScale(kVPrimeScales);
        TLOAD<VPrimeScaleMatrix, 1>(vPrimeScale, gVPrimeScale);

        QTile q;
        QScaleTile qScale;
        QSlice gQ(const_cast<__fp4_e2m1x2 *>(qPtr) + qb * kGroupM * kStoredQD);
        auto gQS = qScaleIter(qb, 0);
        TLOAD<QMatrix, 1>(q, gQ);
        TLOAD<QScaleMatrix, 1>(qScale, gQS);

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
            KTile k;
            KScaleTile kScale;
            KSlice gK(const_cast<__fp4_e2m1x2 *>(kPtr) + kb * kTk * kStoredQD);
            auto gKS = kScaleIter(kb, 0);
            TLOAD<KMatrix, 1>(k, gK);
            TLOAD<KScaleMatrix, 1>(kScale, gKS);

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
            TMATMUL_MX<3>(score, q, qScale, k, kScale, qkOptions);

            // --- online row max on the ln2 lattice ---------------------------
            // 【本段总览】这一大段是算法 B 的在线 softmax + MXFP4 量化，记号为：
            //   s   = 本 kb 块的 QK 分数（bf16）
            //   R   = 到本 kb 为止的在线行 max（每行一个值）
            //   G   = 每组 32 列内的组 max
            //   t   = exp(s - R)              : softmax 分子（本行相对 max）——只是下文
            //                                    推导用的“记法”，kernel 不物化它：
            //                                    实际由 groupExpValues*groupScaleValues 隐含得到，见分母处
            //   m   = exp(G - R) = 组内 t 峰值: 组内最大概率
            //   groupExpValues = 4*exp(s - G) : 喂 fp4 的数据（峰值恒为 4，不饱和）
            //   groupScaleValues = exp(G - R - ln4) = m/4 : 喂 E8M0 的 scale
            // 本段先算 R，再把 R 吸到 ln2 的整数倍（"lattice"），这样在线重标定
            // oldScale = exp(旧R - 新R) 恰好是 2 的整数次幂，可以直接搭 Cube 的
            // CScale（C *= 2^-CScale）省掉一次整 tile 的乘法。
            // 全部在 BF16 域做（用户接受由此带来的精度损失）：不再把行 max 拓宽到
            // FP32 做 `/ln2 → floor → *ln2`，而是 bf16 直接算。代价是 R 不再是精确的
            // ln2 格点，但 CScale 仍由同一差分导出（用于两个累加器的 online 重标定）。
            CScaleTile cscale;          // TMATMUL_MX_ACC 的 CScale（U8 指数）
            Bf16RowValueTile tQ, tRg, tNewMaxB, tDiff;
            IntRowTile tIdx;
            // 1) 本块 QK 的 fused 行 max（PTO #346：已是 final-D 的 bf16 值）直接用
            // 2) tQ = R/ln2
            TMULS(tQ, tLocalMaxB, static_cast<__bf16>(kInvLn2));
            TCVT<LINX_RDN>(tIdx, tQ);                 // floor(R / ln2)
            // 3) tRg = floor(R/ln2)*ln2：把 R 吸到 ln2 格点上（≤ R），全程 bf16
            TCVT(tRg, tIdx);
            TMULS(tRg, tRg, static_cast<__bf16>(kLn2));  // R snapped to the lattice
            if (kb == 0) {
                // 第一块：在线行 max 直接初始化为本块格点值。第一块 PV 走非 ACC 的
                // TMATMUL_MX（直接写出 D，不读旧 C），故不需要预置 cscale=0。
                runningMaxGrid = tRg;
            } else {
                // 后续块：newR = max(旧R, 本块R)
                TMAX(tNewMaxB, runningMaxGrid, tRg);
                TROWEXPANDEXPDIF(oldScale, runningMaxGrid, tNewMaxB);
                // CScale = (新R - 旧R)/ln2，取整；C *= 2^-CScale 即 online 重标定
                TSUB(tDiff, tNewMaxB, runningMaxGrid);
                TMULS(tDiff, tDiff, static_cast<__bf16>(kInvLn2));
                TCVT(tIdx, tDiff);
                TCVT(cscale, tIdx);                   // U8 raw exponent
                runningMaxGrid = tNewMaxB;            // 更新在线行 max
            }
            // runningMaxGrid 即本块更新后的行 max，下面 exp(s - R) 直接用它。

            // --- per-group MXFP4 quantization (per 32-column group) --------------
            // 把 [M, kTk] 的 score 沿 K 切成 kPScaleCols 个 32 列一组（一个 MX group）
            auto blocks =
                TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(score);
            Bf16GroupMaxTile groupLogReference;
            TSUBS(groupLogReference, tGroupMaxB, static_cast<__bf16>(kLn4));
            // P is a read-only assembled parent shared by the two Cube
            // consumers (PV' and PV).
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            // A BF16 M32 CELL contains two group maxima. Bind that CELL,
            // then select the group using its byte offset (0 or 2).
            using Bf16GroupPair = VecTileM32<__bf16, kPeM, 2>;
            auto referencePairs =
                TPARTVIEW<Bf16GroupPair, 1, kPScaleCols / 2>(groupLogReference);
            auto quantizeGroup = [&]<size_t Group>() __attribute__((always_inline)) {
                auto scoreView = blocks[0][Group];
                auto referenceView = referencePairs[0][Group / 2];
                Bf16ScoreGroupTile groupExpValues;
                TROWEXPANDEXPDIF<Bf16ScoreGroupTile,
                    Bf16ScoreTile, Bf16ScoreGroupTile,
                    Bf16GroupMaxTile, Bf16GroupPair, 2 * (Group % 2)>(
                        groupExpValues, scoreView, referenceView);
                TCVT(pFragments[0][Group], groupExpValues);
            };
            [&]<size_t... Groups>(std::index_sequence<Groups...>) __attribute__((always_inline)) {
                (quantizeGroup.template operator()<Groups>(), ...);
            }(std::make_index_sequence<kPScaleCols>{});
            P p = TASSEMBLY<P>(std::move(pFragments));

            // Convert all groupScaleValues to E8M0 scales in one instruction.
            // Keeping the four valid scales in a single [M,G] CELL gives the
            // contiguous-valid layout TMATMUL_MX expects (see the PScale note).
            // 一次性给所有组算 E8M0 scale：E = floor_pow2(m/4) = floor_pow2(exp(G-R-ln4))
            Bf16GroupMaxTile groupScaleValues;
            PScale pScale;
            TROWEXPANDEXPDIF(groupScaleValues, groupLogReference, runningMaxGrid);
            TCVT<LINX_RDN>(pScale, groupScaleValues);  // E8M0（单 CELL）


            // V' denominator: Cube performs the K reduction and keeps the
            // cross-kb result in the same ordinary TileReg accumulator.
            if (kb == 0) {
                TMATMUL_MX<3>(weightedDenom, p, pScale, vPrime,
                              vPrimeScale, denomOptions, kGroupM);
            } else {
                Fp32RowValueTile oldScaleF32;
                TCVT(oldScaleF32, oldScale);
                TROWEXPANDMUL(weightedDenom, weightedDenom, oldScaleF32);
                TMATMUL_MX_ACC<3>(weightedDenom, weightedDenom,
                                  p, pScale, vPrime, vPrimeScale,
                                  denomOptions, kGroupM);
            }

            // --- PV: TMATMUL_MX_ACC folds add + CScale rescale ----------------
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) + kb * kStoredTk * vD);
            // Each kb block owns kPScaleCols consecutive columns of the
            // row-major [vD, Skv/32] ScaleB bin.
            GmVScale gVS(vScalePtr + kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
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

        // Normalize the Cube PV result by the Cube P x V' denominator.
        Fp32RowValueTile runningSumF32;
        // Rescale both FP32 columns throughout the loop, then consume column 0.
        using DenomColumn = CubeAccumulatorM32<float, kPeM, 1>;
        auto denomColumns = TPARTVIEW<DenomColumn, 1, 2>(weightedDenom);
        auto denomFirst = denomColumns[0][0];
        TCVT(runningSumF32, denomFirst);
        TROWEXPANDDIV(weightedValueSum, weightedValueSum, runningSumF32);
        Bf16StoreTile oBf16;
        TCVT(oBf16, weightedValueSum);
        auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
        TSTORE_CUBE(gOut, oBf16);
    }
}
}  // namespace fa_lowp_logdomain
