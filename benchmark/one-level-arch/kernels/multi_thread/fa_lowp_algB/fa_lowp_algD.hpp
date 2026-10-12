#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

#ifndef FA_ALGB_D_PRELOAD
#define FA_ALGB_D_PRELOAD 0
#endif
#ifndef FA_ALGB_D_PRELOAD_N
#define FA_ALGB_D_PRELOAD_N 0
#endif
#ifndef FA_ALGB_D_DBLQ
#define FA_ALGB_D_DBLQ 0
#endif
#ifndef FA_ALGB_D_CONST_KV_ADDR
#define FA_ALGB_D_CONST_KV_ADDR 0
#endif

static_assert(!FA_ALGB_D_CONST_KV_ADDR || FA_ALGB_D_DBLQ,
              "constant KV addresses require Algo D dblQ");
static_assert(!FA_ALGB_D_PRELOAD ||
                  (FA_ALGB_D_PRELOAD_N >= 2 && FA_ALGB_D_PRELOAD_N <= 5),
              "Algo D preload count must be 2..5");

// Algorithm D: log-domain softmax + MXFP4 quantization.
// Denominator: independent Vector sums per group, rescaled against each
// group's current log reference; align and combine the groups on the final KV block.
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

#ifdef FA_LOWP_DEBUG
inline uint8_t functionalDebug[Tsq / 128][Tskv / Tk][16][128 * 128 * 4];
template <typename T>
__attribute__((always_inline)) void dump_functional(const T &tile, int stage, int qb, int kb, int group = -1) {
    unsigned tid = get_thread_idx();
    auto *data = reinterpret_cast<typename T::DType *>(functionalDebug[qb][kb][stage + (group < 0 ? 0 : group)]);
    global_tensor<typename T::DType, RowMajor<T::ValidRow, T::ValidCol>> gm(data + tid * T::ValidRow * T::ValidCol);
    TSTORE_CUBE(gm, tile);
}
#define FA_DUMP(tile, name, ...) dump_functional(tile, name, __VA_ARGS__)
#else
#define FA_DUMP(tile, name, ...) ((void)0)
#endif

// GroupMax comes directly from the fused BF16 CUBE CELL output (GM2).

constexpr int kPeNum = 4;
constexpr int kMxGroup = 32;
constexpr int kPackedFactor = 2;
constexpr float kLn4 = 1.3862943611198906f;    // ln(4)
constexpr float kInvLn2 = 1.4426950408889634f;  // 1/ln(2)
constexpr float kLn2 = 0.6931471805599453f;     // ln(2)

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk>
void fa_lowp_algD_impl(__bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
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
    constexpr int kQBatch = FA_ALGB_D_DBLQ ? 2 : 1;
    static_assert(kQBlocks % kQBatch == 0,
                  "dblQ requires pairs of distinct 128-row Q tiles");
    constexpr int kKVBlocks = Skv / kTk;
    // Continuous lookahead: kPreloadCount is the in-flight distance N (how many
    // KV blocks ahead K/KScale are fetched); one extra slot lets iteration kb
    // issue block kb+N into the slot consumed at kb-1 without a WAR stall.
    constexpr int kPreloadCount =
        FA_ALGB_D_PRELOAD ? (FA_ALGB_D_PRELOAD_N < kKVBlocks
                                 ? FA_ALGB_D_PRELOAD_N
                                 : kKVBlocks)
                          : 0;
    constexpr int kPreloadDepth = kPreloadCount + (FA_ALGB_D_PRELOAD ? 1 : 0);
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
            TEXPANDS(runningMaxGrids[qi], static_cast<__bf16>(-1.0e30f));
            TEXPANDS(runningSums[qi], static_cast<__bf16>(0.0f));
            const int qb = qbBase + qi;
            QSlice gQ(const_cast<__fp4_e2m1x2 *>(qPtr) + qb * kGroupM * kStoredQD);
            auto gQS = qScaleIter(qb, 0);
            TLOAD<QMatrix, 1>(queries[qi], gQ);
            TLOAD<QScaleMatrix, 1>(queryScales[qi], gQS);
        }

#if FA_ALGB_D_PRELOAD
        // Continuous lookahead ring of K/KScale tiles.
        //   slot s holds block b  <=>  b % kPreloadDepth == s.
        // Tile lifetime: the prologue fills blocks 0..N-1; iteration kb reads
        // slot kb%depth (block kb) and, before doing any of block kb's scalar
        // softmax work, issues the load for block kb+N into slot (kb+N)%depth.
        // That slot last held block kb+N-depth = kb-1, already consumed by the
        // previous iteration's QK, so the overwrite is safe and its TLOAD is
        // not serialized behind this block's scalar address/softmax stream.
        KTile kRing[kPreloadDepth];
        KScaleTile kScaleRing[kPreloadDepth];
#pragma clang loop unroll(full)
        for (int preload = 0; preload < kPreloadCount; ++preload) {
            // With dblQ+preload, the ring still carries distinct lookahead
            // slots, but constKV mode intentionally removes the per-slot
            // address arithmetic and reads block 0 for every slot.
#if FA_ALGB_D_CONST_KV_ADDR
            KSlice gKPre(const_cast<__fp4_e2m1x2 *>(kPtr));
            auto gKSPre = kScaleIter(0, 0);
#else
            KSlice gKPre(const_cast<__fp4_e2m1x2 *>(kPtr) +
                         preload * kTk * kStoredQD);
            auto gKSPre = kScaleIter(preload, 0);
#endif
            TLOAD<KMatrix, 1>(kRing[preload], gKPre);
            TLOAD<KScaleMatrix, 1>(kScaleRing[preload], gKSPre);
        }
#endif

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
#if FA_ALGB_D_PRELOAD
            KTile &kUse = kRing[kb % kPreloadDepth];
            KScaleTile &kScaleUse = kScaleRing[kb % kPreloadDepth];
            // Lookahead is a compile-time-described constant block for the
            // fully unrolled loop; the last kPreloadCount iterations have none.
            if (kb + kPreloadCount < kKVBlocks) {
                KTile &kNext = kRing[(kb + kPreloadCount) % kPreloadDepth];
                KScaleTile &kScaleNext =
                    kScaleRing[(kb + kPreloadCount) % kPreloadDepth];
#if FA_ALGB_D_CONST_KV_ADDR
                KSlice gKNext(const_cast<__fp4_e2m1x2 *>(kPtr));
                auto gKSNext = kScaleIter(0, 0);
#else
                KSlice gKNext(const_cast<__fp4_e2m1x2 *>(kPtr) +
                              (kb + kPreloadCount) * kTk * kStoredQD);
                auto gKSNext = kScaleIter(kb + kPreloadCount, 0);
#endif
                TLOAD<KMatrix, 1>(kNext, gKNext);
                TLOAD<KScaleMatrix, 1>(kScaleNext, gKSNext);
            }
#else
            KTile k;
            KScaleTile kScale;
#if FA_ALGB_D_CONST_KV_ADDR
            // Timing-only: every kb reads block 0 to remove KV address updates.
            KSlice gK(const_cast<__fp4_e2m1x2 *>(kPtr));
            auto gKS = kScaleIter(0, 0);
#else
            KSlice gK(const_cast<__fp4_e2m1x2 *>(kPtr) + kb * kTk * kStoredQD);
            auto gKS = kScaleIter(kb, 0);
#endif
            TLOAD<KMatrix, 1>(k, gK);
            TLOAD<KScaleMatrix, 1>(kScale, gKS);
#endif

#if FA_ALGB_D_DBLQ
            // Load K/V only once per pair of different Q tiles.
            VTile v;
            VScaleTile vScale;
#if FA_ALGB_D_CONST_KV_ADDR
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr));
            GmVScale gVS(vScalePtr);
#else
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) + kb * kStoredTk * vD);
            GmVScale gVS(vScalePtr + kb * kPScaleCols);
#endif
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
#endif
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
#if FA_ALGB_D_PRELOAD
                TMATMUL_MX<3>(score, q, qScale, kUse, kScaleUse, qkOptions);
#else
                TMATMUL_MX<3>(score, q, qScale, k, kScale, qkOptions);
#endif
                FA_DUMP(score, 0, qb, kb);
                FA_DUMP(tGroupMaxB, 1, qb, kb);

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
                Bf16GroupMaxTile groupUpdateScale, groupScaleValues;
                if (kb != 0)
                    TEXPDIF(groupUpdateScale, previousGroupLogReference,
                            groupLogReference);
                TROWEXPANDEXPDIF(groupScaleValues, groupLogReference, runningMaxGrid);
                FA_DUMP(runningMaxGrid, 2, qb, kb);
                FA_DUMP(groupLogReference, 3, qb, kb);
                FA_DUMP(groupScaleValues, 4, qb, kb);
                // Assemble the group fragments into the read-only PV input.
                TileArray<PBlock, 1, kPScaleCols> pFragments;
                // Finish denominator work before opening the assembled P parent.
                Bf16ScoreGroupTile groupExpTiles[kPScaleCols];
                Bf16RowValueTile localSum;
                if (kb == kKVBlocks - 1)
                    TEXPANDS(localSum, static_cast<__bf16>(0.0f));
                // A BF16 CELL contains two groups; byte offset selects the slot.
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
                    FA_DUMP(groupExpValues, 5, qb, kb, Group);
                    WideBf16GroupReductionTile sumWide;
                    TROWSUM(sumWide, groupExpValues);
                    auto groupExpSum = TREDUCEPREFIXVIEW<Bf16RowValueTile>(sumWide);
                    FA_DUMP(sumWide, 9, qb, kb, Group);
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
                // Keeping the four valid scales in a single [M,G] CELL gives the
                // contiguous-valid layout TMATMUL_MX expects (see the PScale note).
                // 一次性给所有组算 E8M0 scale：E = floor_pow2(m/4) = floor_pow2(exp(G-R-ln4))
                PScale pScale;
                TCVT<LINX_RDN>(pScale, groupScaleValues);  // E8M0（单 CELL）

                if (kb == kKVBlocks - 1) runningSum = localSum;
                FA_DUMP(runningSum, 13, qb, kb);


                // --- PV: TMATMUL_MX_ACC folds add + CScale rescale ----------------
#if !FA_ALGB_D_DBLQ
                VTile v;
                VScaleTile vScale;
                VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) + kb * kStoredTk * vD);
                // Each kb block owns kPScaleCols consecutive columns of the
                // row-major [vD, Skv/32] ScaleB bin.
                GmVScale gVS(vScalePtr + kb * kPScaleCols);
                TLOAD<VMatrix, 1>(v, gV);
                TLOAD<VScaleMatrix, 1>(vScale, gVS);
#endif
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
            FA_DUMP(weightedValueSum, 14, qb, kKVBlocks - 1);
            TCVT(runningSumF32, runningSum);
            TROWEXPANDDIV(weightedValueSum, weightedValueSum, runningSumF32);
            Bf16StoreTile oBf16;
            TCVT(oBf16, weightedValueSum);
            auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
            TSTORE_CUBE(gOut, oBf16);
        }
    }
}
}  // namespace fa_lowp_logdomain
