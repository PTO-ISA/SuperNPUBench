#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

// fa_lowp log-domain softmax + MXFP4 quantization — algorithms B/C.
// =====================================================================
// Per 32-column MX group (no algorithm-A / no debug branches):
//
//   G   = groupmax(s)                     (fused B.FPATR GroupMax, final BF16 D)
//   P3  = G - ln4
//   P4  = exp(s - P3) = 4*exp(s - G)      (data; group peak exactly 4 -> no E2M1 sat)
//   P5  = exp(P3 - R) = exp(G - R - ln4)  (scale before E8M0; R = online row max)
//   E   = E8M0_floor(P5)
//   fp4 = E2M1(P4)
//
// Dequant: fp4 * E = t * floor_pow2(m)/m,  m = exp(G - R),  t = exp(s - R).
//
// Online accumulation: FP32 PV accumulator whose add AND online rescale are
// folded into `TMATMUL_MX_ACC` via the Cube CScale (C *= 2^-CScale).  For that
// the row max is kept on the ln2 lattice so the rescale factor is an exact
// power of two.
namespace fa_lowp_logdomain {
using namespace pto;

// Algorithm selector:
//   FA_ALGB_ALGO_C=1 : V' denominator uses Cube P×V' with ordinary TileReg
//                        accumulation across kb; online rescale is applied
//                        directly to the Cube denominator tile.
//   FA_ALGB_ALGO_C=0 : legacy algorithm-B denominator path.
#ifndef FA_ALGB_ALGO_C
#define FA_ALGB_ALGO_C 1
#endif

// 组 max 来源（设计意图）：
//   FA_ALGB_GM_FUSED=2 : Algo C direct BF16 CELL/byte-slot broadcast (PTO #358).
//                        Requires the dual-subview/valid-column TileOP fix.
//   FA_ALGB_GM_FUSED=1 : 直接使用 fixp/CUBE 产生的 GroupMaxOut；每个 group
//                        的值作为 TROWEXPANDEXPDIF 的 row-broadcast 输入，不能
//                        额外重算 TROWMAX。
//   FA_ALGB_GM_FUSED=0 : 当前 gfsim 可验证的 fallback，会用 TROWMAX 重算；
//                        这不是最终数据流，只是绕开 BF16x2 CellReg/subview 限制。
#ifndef FA_ALGB_GM_FUSED
#define FA_ALGB_GM_FUSED 0
#endif
static_assert(FA_ALGB_GM_FUSED >= 0 && FA_ALGB_GM_FUSED <= 2);
static_assert(FA_ALGB_GM_FUSED != 2 || FA_ALGB_ALGO_C,
              "direct GroupMax mode currently requires Algo C");

constexpr int kPeNum = 4;
constexpr int kMxGroup = 32;
constexpr int kPackedFactor = 2;
constexpr float kLn4 = 1.3862943611198906f;    // ln(4)
constexpr float kInvLn2 = 1.4426950408889634f;  // 1/ln(2)
constexpr float kLn2 = 0.6931471805599453f;     // ln(2)

template <typename Out, typename Parent, typename Sub>
inline void materialize_subview(Out &dst, region::SubTileView<Parent, Sub> &src,
                                typename Sub::DType scalar) {
    TMULS(dst, src, scalar);
}

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void fa_lowp_algB_impl(__bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
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
    static_assert(FA_ALGB_GM_FUSED != 2 || kTk == 128,
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
    // The per-group [M,1] fragment mirrors the Fp32RowTile geometry.
    // Fused GroupMaxOut (final D dtype).  The intended data path consumes this
    // [M,G] BF16 tile directly for each group's TROWEXPANDEXPDIF input.  The
    // fallback path below recomputes group max with TROWMAX only because the
    // current BF16x2 CellReg/subview contract cannot expose one group cleanly.
    using Bf16GroupMaxTile = VecTileM32<__bf16, kPeM, kPScaleCols>;
    // FP32 carrier for the fused GroupMaxOut so a per-group TPARTVIEW can
    // isolate one group (FP32 CUBE_M32 CELL = 1 column; bf16 CELL = 2 columns,
    // which would pack two groups per CELL and block the partition).
    using Fp32GroupMaxTile = VecTileM32<float, kPeM, kPScaleCols>;
    using Fp32RowTile = VecTileM32<float, kPeM, 1, kPeM, 1>;
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
#if FA_ALGB_ALGO_C
        Fp32DenomTile weightedDenom;
        TEXPANDS(weightedDenom, 0.0f);
        // V' is GM ND -> SharedTReg ND.  It is read by each non-ACC P×V'
        // instruction, but never retained in internalACC.
        static __bf16 kVPrimeOnes[kTk];
#pragma clang loop unroll(full)
        for (int i = 0; i < kTk; ++i)
            kVPrimeOnes[i] = static_cast<__bf16>(1.0f);
        VPrimeTile vPrime;
        VPrimeGm gVPrime(kVPrimeOnes);
        TLOAD<VPrimeMatrix, 1>(vPrime, gVPrime);
#endif

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
            //                                    实际由 P4*P5 隐含得到，见分母处
            //   m   = exp(G - R) = 组内 t 峰值: 组内最大概率
            //   P4  = 4*exp(s - G)            : 喂 fp4 的数据（峰值恒为 4，不饱和）
            //   P5  = exp(G - R - ln4) = m/4  : 喂 E8M0 的 scale
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

            // --- algorithm B quantization (per 32-column group) --------------
            // 把 [M, kTk] 的 score 沿 K 切成 kPScaleCols 个 32 列一组（一个 MX group）
            auto blocks =
                TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(score);
            Bf16GroupMaxTile p3Group;
            TSUBS(p3Group, tGroupMaxB, static_cast<__bf16>(kLn4));
#if FA_ALGB_GM_FUSED == 1
            // 组 max 取 fused GroupMaxOut：bf16 [M,G] 拓成 fp32（fp32 CELL=1 列，
            // TPARTVIEW 才能把单组干净切开；bf16→fp32 无损），再降回 bf16。
            Fp32GroupMaxTile gMaxF32;
            TCVT(gMaxF32, tGroupMaxB);
            auto gmaxBlocks = TPARTVIEW<Fp32RowTile, 1, kPScaleCols>(gMaxF32);
#endif
            // P is a read-only assembled parent shared by the two Cube
            // consumers (PV' and PV).
            TileArray<PBlock, 1, kPScaleCols> pFragments;
#if !FA_ALGB_ALGO_C
            Bf16RowValueTile localSum;
            TEXPANDS(localSum, static_cast<__bf16>(0.0f));
#endif
#if FA_ALGB_GM_FUSED == 2
            // A BF16 M32 CELL contains two group maxima. Bind that CELL,
            // then select the group using its byte offset (0 or 2).
            using Bf16GroupPair = VecTileM32<__bf16, kPeM, 2>;
            auto groupPairs =
                TPARTVIEW<Bf16GroupPair, 1, kPScaleCols / 2>(p3Group);
            auto quantizeGroup = [&]<size_t Group>() {
                auto scoreView = blocks[0][Group];
                auto maxView = groupPairs[0][Group / 2];
                Bf16ScoreGroupTile p4;
                TROWEXPANDEXPDIF<Bf16ScoreGroupTile,
                    Bf16ScoreTile, Bf16ScoreGroupTile,
                    Bf16GroupMaxTile, Bf16GroupPair, 2 * (Group % 2)>(
                        p4, scoreView, maxView);
                TCVT(pFragments[0][Group], p4);
            };
            [&]<size_t... Groups>(std::index_sequence<Groups...>) {
                (quantizeGroup.template operator()<Groups>(), ...);
            }(std::make_index_sequence<kPScaleCols>{});
#else
#pragma clang loop unroll(full)
            for (int block = 0; block < kPScaleCols; ++block) {
                auto sView = blocks[0][block];

                // P3 = G - ln4
                Bf16RowValueTile p3;
#if FA_ALGB_GM_FUSED
                // G 取 fused GroupMaxOut（fp32 载体逐组切开）
                Fp32RowTile gF32;
                auto gView = gmaxBlocks[0][block];
                materialize_subview(gF32, gView, 1.0f);
                Bf16RowValueTile gB;
                TCVT(gB, gF32);
                TSUBS(p3, gB, static_cast<__bf16>(kLn4));
#else
                // G 用向量 TROWMAX 重算（gfrun/gfsim 均可）
                WideBf16GroupReductionTile gWide;
                TROWMAX(gWide, sView);
                auto gView = TREDUCEPREFIXVIEW<Bf16RowValueTile>(gWide);
                TSUBS(p3, gView, static_cast<__bf16>(kLn4));
#endif
                // P4 = exp(s - P3) = 4*exp(s - G)  ->  fp4 = E2M1(P4)
                // P4 是量化数据：组内峰值恰为 4，不会碰到 E2M1 的 6 上限
                Bf16ScoreGroupTile p4;
                TROWEXPANDEXPDIF(p4, sView, p3);
                TCVT(pFragments[0][block], p4);
#if !FA_ALGB_ALGO_C
                Bf16RowValueTile p5;
                TROWEXPANDEXPDIF(p5, p3, runningMaxGrid);
                WideBf16GroupReductionTile sumWide;
                TROWSUM(sumWide, p4);
                auto sum4 = TREDUCEPREFIXVIEW<Bf16RowValueTile>(sumWide);
                Bf16RowValueTile contrib;
                TMUL(contrib, sum4, p5);
                TADD(localSum, localSum, contrib);
#endif
            }
#endif
            P p = TASSEMBLY<P>(std::move(pFragments));

            // P scale E8M0 = floor_pow2(exp(P3 - R)) for every group at once.
            // Keeping the four valid scales in a single [M,G] CELL gives the
            // contiguous-valid layout TMATMUL_MX expects (see the PScale note).
            // 一次性给所有组算 E8M0 scale：E = floor_pow2(m/4) = floor_pow2(exp(G-R-ln4))
            Bf16GroupMaxTile p5Group;
            PScale pScale;
            TROWEXPANDEXPDIF(p5Group, p3Group, runningMaxGrid);            // 所有组的 m/4
            TCVT<LINX_RDN>(pScale, p5Group);                        // E8M0（单 CELL）

#if !FA_ALGB_ALGO_C
            Bf16RowValueTile newSum;
            if (kb == 0) TADD(newSum, runningSum, localSum);
            else         TFMA(newSum, runningSum, oldScale, localSum);
#endif

            // V' denominator: Cube performs the K reduction and keeps the
            // cross-kb result in the same ordinary TileReg accumulator.
#if FA_ALGB_ALGO_C
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
#endif

            // --- PV: TMATMUL_MX_ACC folds add + CScale rescale ----------------
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) + kb * kStoredTk * vD);
            // Each kb block owns kPScaleCols consecutive columns of the
            // row-major [vD, Skv/32] ScaleB bin.
            GmVScale gVS(vScalePtr + kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
            // PV CCTRL (PTO-ISA 0.58.6 spec#236 / internal-accumulator.asl):
            // acc_hint (CCTRL[1]) is a non-binding explicit-C cache-use/prefetch
            // hint, legal only on ACC forms.  CCTRL[0] (raw_acc) is optional and
            // dropped: keep_acc already publishes raw FP32 D.  The first block
            // is a plain non-ACC TMATMUL_MX that writes D directly (and
            // CCTRL[1] must be 0 on init=1 forms, so no hint there).
            if (kb == 0) {
                TMATMUL_MX<3>(weightedValueSum, p, pScale, v, vScale,
                              pvOptions, kGroupM);
            } else {
                auto pvOptC =
                    fixp::keep_acc().transpose_b().acc_hint().cscale(cscale);
                TMATMUL_MX_ACC<3>(weightedValueSum, weightedValueSum,
                                  p, pScale, v, vScale, pvOptC, kGroupM);
            }

#if !FA_ALGB_ALGO_C
            runningSum = newSum;
#endif

        }

        // Normalize the Cube PV result by the selected B (Vector) / C (Cube) denominator.
        Fp32RowValueTile runningSumF32;
#if FA_ALGB_ALGO_C
        TCVT(runningSumF32, weightedDenom);
#else
        TCVT(runningSumF32, runningSum);
#endif
        TROWEXPANDDIV(weightedValueSum, weightedValueSum, runningSumF32);
        Bf16StoreTile oBf16;
        TCVT(oBf16, weightedValueSum);
        auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
        TSTORE_CUBE(gOut, oBf16);
    }
}
}  // namespace fa_lowp_logdomain
