#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <utility>

// Experimental large-Tk variant of fa_lowp.hpp.
//
// Keep large P-scale packing and long TileArray assembly experiments here so
// the stable four-scale-column kernel remains unchanged.  Tk=512 needs the
// backend Tile clock-hand allocator (enabled by this testcase's Makefile): a
// 16-writer TASSEMBLY cannot keep one Local generation live using only the
// default M-hand allocation sequence.
//
// The PTO ISA permits a 32-KiB per-PE Local destination, but the current
// timing model does not make forward progress when one GMMA TMATMUL_MX writes
// the complete [32,512] BF16 QK shard.  Split only that QK write into two
// [32,256] destinations.  The two halves still form one Tk=512 online-softmax
// update and one Tk=512 PV, so this is not equivalent to changing the outer
// KV blocking back to Tk=256.

// Full MXFP4 FlashAttention
// =========================
//
// Tensor contract:
//   Q: [Sq,  qD]  MXFP4 (E2M1x2 data + one E8M0 scale per 32 K values)
//   K: [Skv, qD]  MXFP4 (E2M1x2 data + one E8M0 scale per 32 K values)
//   V: [Skv, vD]  MXFP4 (E2M1x2 data + one E8M0 scale per 32 K values)
//   O: [Sq,  vD]  BF16
//
// Data flow for one cooperative 128-row Q block:
//
//   1. QK = Q * K^T
//      Q and K remain MXFP4 inputs and are consumed by TMATMUL_MX.  The
//      fixpipe option is deliberately fixp::bf16(), so the matrix result is
//      converted from the internal FP32 accumulator to BF16 before any
//      vector operation.  QK is NOT immediately quantized to FP4.
//
//   2. Online softmax in BF16
//      Scale QK by 1/sqrt(qD), update the running row max, compute exp(),
//      update the running row sum, and rescale the previous PV accumulator
//      when a later KV block raises the row max.
//
//   3. Quantize P only at the PV boundary
//      The BF16 exp result is split into 32-column MX groups.  Each group
//      produces an E2M1x2 data fragment and one E8M0 row scale.  Since exp()
//      is nonnegative, rowmax(P) is already absmax(P); no TABS is needed.
//
//   4. Assemble and multiply P * V
//      TASSEMBLY joins all data fragments into the complete local P tile and
//      independently joins the scale fragments into the complete P-scale
//      matrix.  P and V are then consumed by a second TMATMUL_MX.  Each PV
//      result, the online weighted-value sum, and final normalization all
//      remain BF16; the normalized BF16 Tile is stored directly as O.
//
// Current TileOP API notes and compromises:
//
//   * PTO #311 makes a CUBE row-reduction result logically [M,1] but requires
//     a wide physical carrier matching the source columns.  The updated API's
//     TREDUCEPREFIXVIEW exposes its first [M,1] CELL without a copy; mixed
//     TMAX/TADD overloads consume that view directly, and TFMA consumes the
//     prefix as its fused addend during the online-sum update.
//
//   * Per-group amax uses the same reduction-prefix view. TMULS consumes its
//     prefix directly, avoiding a wide-to-compact reduction TCVT or copy.
namespace fa_lowp_ltile {
using namespace pto;

constexpr int kPeNum = 4;       // One cooperative QK/PV matmul uses four PEs.
constexpr int kMxGroup = 32;    // OCP MX scale granularity along matrix K.
constexpr int kPackedFactor = 2;  // E2M1x2 packs two logical FP4 values/byte.

template <int Word, int WordCount, typename ScaleExponentTile,
          typename ScaleWordTile, typename PackedScaleWordsGm,
          typename QuantizeGroup>
__attribute__((always_inline)) inline void pack_p_scale_words(
    QuantizeGroup &quantizeGroup, uint32_t *packedScaleScratch) {
    constexpr int kFirstGroup = Word * 4;
    ScaleExponentTile scaleCode0, scaleCode1, scaleCode2, scaleCode3;
    quantizeGroup.template operator()<kFirstGroup + 0>(scaleCode0);
    quantizeGroup.template operator()<kFirstGroup + 1>(scaleCode1);
    quantizeGroup.template operator()<kFirstGroup + 2>(scaleCode2);
    quantizeGroup.template operator()<kFirstGroup + 3>(scaleCode3);

    ScaleWordTile scaleWord0, scaleWord1, scaleWord2, scaleWord3;
    TCVT(scaleWord0, scaleCode0);
    TCVT(scaleWord1, scaleCode1);
    TCVT(scaleWord2, scaleCode2);
    TCVT(scaleWord3, scaleCode3);

    ScaleWordTile scalePair01, scalePair23, packedScaleWord;
    TPACK(scalePair01, scaleWord0, scaleWord1, 0x00000101);
    TPACK(scalePair23, scaleWord2, scaleWord3, 0x00000101);
    TPACK(packedScaleWord, scalePair01, scalePair23, 0x00000202);

    PackedScaleWordsGm packedWordsGm(packedScaleScratch + Word);
    TSTORE_CUBE(packedWordsGm, packedScaleWord);

    if constexpr (Word + 1 < WordCount) {
        pack_p_scale_words<Word + 1, WordCount, ScaleExponentTile,
                           ScaleWordTile, PackedScaleWordsGm>(
            quantizeGroup, packedScaleScratch);
    }
}

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD, bool kBf16RecipFromE8M0 = false>
void flash_attention_lowp_impl(
    __bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
    const __fp4_e2m1x2 *kPtr, const __fp4_e2m1x2 *vPtr,
    const __fp8_e8m0 *qScalePtr, const __fp8_e8m0 *kScalePtr,
    const __fp8_e8m0 *vScalePtr) {
    constexpr int kGroupM = kTm <= 128 ? kTm : 128;
    constexpr int kPeM = 32;
    constexpr int kStoredQD = qD / kPackedFactor;
    constexpr int kStoredTk = kTk / kPackedFactor;
    constexpr int kStoredVD = vD / kPackedFactor;
    constexpr int kQScaleCols = qD / kMxGroup;
    constexpr int kPScaleCols = kTk / kMxGroup;
    constexpr int kPScaleWordCols = kPScaleCols / 4;
    constexpr int kPaddedQScaleCols = ((kQScaleCols + 31) / 32) * 32;
    constexpr int kPaddedPScaleCols = ((kPScaleCols + 31) / 32) * 32;
    constexpr int kQBlocks = Sq / kGroupM;
    constexpr int kKVBlocks = Skv / kTk;
    constexpr int kQkHalfTk = kTk / 2;
    constexpr int kPScaleColsPerHalf = kQkHalfTk / kMxGroup;
    const uint32_t tid = get_thread_idx();

    // This first implementation intentionally fixes the cooperative shape to
    // 128 rows: four PEs each own one 32-row CUBE_M32 result shard.
    static_assert(kGroupM == 128,
                  "fa_lowp requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    static_assert((kPScaleCols & (kPScaleCols - 1)) == 0,
                  "P scale packing requires a power-of-two block count");
    static_assert(kPScaleCols >= 4 && kPScaleCols % 4 == 0,
                  "P scale packing requires a multiple of four groups");
    static_assert(kTk % (2 * kMxGroup) == 0,
                  "split QK halves must contain complete MX groups");

    // Packed GM data layouts.  The stored K dimension is divided by two
    // because every E2M1x2 byte contains two adjacent logical K values.
    // V is the physically transposed Shared-B operand [K,N].  Packed-X2
    // ordinary RowMajor storage pairs adjacent columns within each row, so
    // its GM carrier shape is [K,N/2].
    using QSlice = global_tensor<__fp4_e2m1x2,
                                 RowMajor<kGroupM, kStoredQD>>;
    using KHalfSlice = global_tensor<__fp4_e2m1x2,
                                     RowMajor<kQkHalfTk, kStoredQD>>;
    using VSlice = global_tensor<__fp4_e2m1x2,
                                 RowMajor<kTk, kStoredVD>>;
    // MX scale layouts follow the logical matrix-multiply K dimension:
    // Q scale [Sq,qD/32], K scale [Skv,qD/32], V scale [vD,Skv/32].
    using GmQScale = global_tensor<__fp8_e8m0,
                                   RowMajor<Sq, kQScaleCols>>;
    using GmKScale = global_tensor<__fp8_e8m0,
                                   RowMajor<Skv, kQScaleCols>>;
    using GmVScale = global_tensor<__fp8_e8m0,
                                   RowMajor<vD, Skv / kMxGroup>>;
    using GmO = global_tensor<__bf16, RowMajor<Sq, vD>>;

    // Shared matrix extents use logical element counts even for packed FP4;
    // only GM pointer arithmetic uses the divided-by-two carrier counts.
    //
    // QK: Shared B without TransB declares physical [N,K], hence KMatrix is
    // [kTk,qD].  PV: V arrives as physical [K,N], hence pvOptions below sets
    // TransB and VMatrix is [kTk,vD].
    using QMatrix = SharedMatrixLeft<__fp4_e2m1x2, 128, qD, 128, qD>;
    using KHalfMatrix =
        SharedMatrixRight<__fp4_e2m1x2, kQkHalfTk, qD>;
    using VMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, vD>;
    using QTile = SharedTile<QMatrix>;
    using KHalfTile = SharedTile<KHalfMatrix>;
    using VTile = SharedTile<VMatrix>;

    // Shared scale tiles are padded to the minimum legal physical matrix
    // carrier while ValidCol/ValidRow describe only the real group-32 scales.
    // TMATMUL_MX validates and consumes the valid shapes, not the padding.
    using QScaleMatrix = SharedMatrixLeft<
        __fp8_e8m0, 128, kPaddedQScaleCols, 128, kQScaleCols>;
    using KHalfScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, kQkHalfTk, kPaddedQScaleCols,
        kQkHalfTk, kQScaleCols>;
    using VScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, vD, kPaddedPScaleCols, vD, kPScaleCols>;
    using QScaleTile = SharedTile<QScaleMatrix>;
    using KHalfScaleTile = SharedTile<KHalfScaleMatrix>;
    using VScaleTile = SharedTile<VScaleMatrix>;

    // Per-PE QK/softmax tiles.  A reduction carrier keeps the source's
    // physical column span but has only one valid value per row (PTO #311).
    // A row-value tile is the compact, one-CELL form of those row scalars:
    // one valid column, two physical BF16 columns in CUBE_M32.
    using Bf16ScoreHalfTile = CubeTileM32<__bf16, kPeM, kQkHalfTk>;
    using WideBf16RowReductionTile =
        VecTileM32<__bf16, kPeM, kQkHalfTk, kPeM, 1>;
    using Bf16RowValueTile = VecTileM32<__bf16, kPeM, 2, kPeM, 1>;
    using Bf16ScoreGroupTile = CubeTileM32<__bf16, kPeM, kMxGroup>;
    using WideBf16GroupReductionTile =
        VecTileM32<__bf16, kPeM, kMxGroup, kPeM, 1>;
    // P is quantized group-by-group.  PBlock contains kMxGroup logical E2M1
    // values per row; each group produces one E8M0 scale code per row.
    // __fp4_e2m1x2 already packs two logical 4-bit values in one byte, so the
    // Tile column count remains the logical column count and must not be
    // doubled to account for the carrier representation.
    //
    // Every four 32x1 raw scale-code columns are packed byte-wise into one
    // 32x1 U32 CUBE_M32 cell.  The packed words are stored with row stride
    // kPScaleWordCols and reloaded as the byte-equivalent compact
    // [32,kPScaleCols] E8M0 scale tile required by TMATMUL_MX.  TASSEMBLY
    // cannot concatenate the raw U8 fragments directly: each fragment is
    // padded to 128 B, which would place live columns at 0/4/8/12...
    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    using PScaleExponentU8 = Tile<Location::Scaling, uint8_t,
        kPeM, 4, BLayout::CubeM32, kPeM, 1>;
    using PScaleWordU32 = Tile<Location::Scaling, uint32_t,
        kPeM, 1, BLayout::CubeM32, kPeM, 1>;
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, kPScaleCols, BLayout::CubeM32, kPeM, kPScaleCols>;
    using GmPackedPScaleWords =
        global_tensor<uint32_t, RowMajor<kPeM, kPScaleWordCols>>;
    using GmPackedPScaleE8M0 =
        global_tensor<__fp8_e8m0, RowMajor<kPeM, kPScaleCols>>;
    using Bf16WeightedValueTile = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using Bf16PvTile = Bf16WeightedValueTile;

    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KHalfScaleIter = global_iterator<GmKScale, KHalfScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16WeightedValueTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KHalfScaleIter kHalfScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    OIter outIter(outPtr);

    // Shared Q/K and their scales coexist during QK; V and its scale coexist
    // during PV.  Each phase must fit the 256-KiB cooperative SharedTReg pool.
    static_assert(QMatrix::LogicalTileBytes +
                      2 * KHalfMatrix::LogicalTileBytes +
                      QScaleMatrix::LogicalTileBytes +
                      2 * KHalfScaleMatrix::LogicalTileBytes <= 256 * 1024);
    static_assert(VMatrix::LogicalTileBytes +
                      VScaleMatrix::LogicalTileBytes <= 256 * 1024);
    static_assert(P::LogicalTileBytes ==
                  PBlock::LogicalTileBytes * kPScaleCols);
    static_assert(PScale::LogicalTileBytes ==
                      PScaleWordU32::LogicalTileBytes * kPScaleWordCols,
                  "packed U32 words and E8M0 scale tile must share storage");

    // TPACK is specified to produce a U32 Tile.  A C++ bit_cast/reinterpret
    // changes only the source-level type and does not retag the architectural
    // Tile-register descriptor, so TMATMUL_MX cannot consume that register as
    // E8M0 directly.  Use one private 128 B stack slot as a storage-preserving
    // U32 -> E8M0 retag bridge until TileOP provides a cross-element-width
    // storage reinterpret operation.
    alignas(128) uint32_t packedPScaleScratch[kPeM * kPScaleWordCols];

    // E2M1 has maximum exponent emax=2.  MX scaling is based on the exponent
    // bound 2^emax=4, not on the maximum finite E2M1 value 6:
    //   scale = 2^floor(log2(amax)) / 4.
    // Multiplying by 1/4 and converting to E8M0 with RTM implements this.
    const __bf16 qkScale = static_cast<__bf16>(
        1.0f / sqrt(static_cast<float>(scaleD)));
    const __bf16 kInvFp4PowerMax = static_cast<__bf16>(0.25f);
    constexpr auto qkOptions = fixp::bf16();
    constexpr auto pvOptions = fixp::bf16().transpose_b();

#pragma clang loop unroll(full)
    for (int qb = 0; qb < kQBlocks; ++qb) {
        // Online softmax: runningSum is the probability denominator, while
        // weightedValueSum accumulates the unnormalized sum of P * V in BF16.
        // The running max and denominator also stay BF16.
        Bf16WeightedValueTile weightedValueSum;
        Bf16RowValueTile runningMax, runningSum;
        TEXPANDS(runningMax, static_cast<__bf16>(-1.0e30f));
        TEXPANDS(runningSum, static_cast<__bf16>(0.0f));

        // Q and its scales are invariant across all KV blocks for this query
        // block, so load them once outside the inner loop.
        QTile q;
        QScaleTile qScale;
        QSlice gQ(const_cast<__fp4_e2m1x2 *>(qPtr) +
                  qb * kGroupM * kStoredQD);
        auto gQS = qScaleIter(qb, 0);
        TLOAD<QMatrix, 1>(q, gQ);
        TLOAD<QScaleMatrix, 1>(qScale, gQS);

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
            // Load the two physical halves of this logical Tk=512 K block.
            // Each half has its own Shared generation and produces a 16-KiB
            // per-PE BF16 score tile.  The full K/V blocking remains 512.
            KHalfTile kLo, kHi;
            KHalfScaleTile kScaleLo, kScaleHi;
            KHalfSlice gKLo(const_cast<__fp4_e2m1x2 *>(kPtr) +
                            kb * kTk * kStoredQD);
            KHalfSlice gKHi(const_cast<__fp4_e2m1x2 *>(kPtr) +
                            kb * kTk * kStoredQD +
                            kQkHalfTk * kStoredQD);
            auto gKSLo = kHalfScaleIter(kb * 2, 0);
            auto gKSHi = kHalfScaleIter(kb * 2 + 1, 0);
            TLOAD<KHalfMatrix, 1>(kLo, gKLo);
            TLOAD<KHalfScaleMatrix, 1>(kScaleLo, gKSLo);
            TLOAD<KHalfMatrix, 1>(kHi, gKHi);
            TLOAD<KHalfScaleMatrix, 1>(kScaleHi, gKSHi);

            // QK matrix stage.  Splitting the destination avoids the current
            // timing-model deadlock on one [32,512] BF16 Local write.  Both
            // halves use the same Q tile and are combined before the online
            // softmax state is updated.
            Bf16ScoreHalfTile probabilityLo, probabilityHi;
            TMATMUL_MX<3>(probabilityLo, q, qScale, kLo, kScaleLo,
                          qkOptions);
            TMATMUL_MX<3>(probabilityHi, q, qScale, kHi, kScaleHi,
                          qkOptions);
            TMULS(probabilityLo, probabilityLo, qkScale);
            TMULS(probabilityHi, probabilityHi, qkScale);

            // Row max for numerically stable online softmax.  PTO #311 keeps
            // each [32,1] result in the prefix CELL of a wide carrier.  First
            // merge the two QK halves, then merge with the running maximum.
            WideBf16RowReductionTile localMaxLoWide, localMaxHiWide;
            TROWMAX(localMaxLoWide, probabilityLo);
            TROWMAX(localMaxHiWide, probabilityHi);
            auto localMaxLo =
                TREDUCEPREFIXVIEW<Bf16RowValueTile>(localMaxLoWide);
            auto localMaxHi =
                TREDUCEPREFIXVIEW<Bf16RowValueTile>(localMaxHiWide);
            Bf16RowValueTile localMax;
            TMAX(localMax, localMaxLo, localMaxHi);
            // Online max update.  If the max changes, oldScale computes
            // exp(runningMax-newMax) and rescales the already accumulated PV
            // contribution.  Both the row scale and weighted-value sum are
            // BF16, so no dtype conversion is needed before broadcasting.
            Bf16RowValueTile newMax, oldScale;
            TMAX(newMax, runningMax, localMax);
            if (kb != 0) {
                TROWEXPANDEXPDIF(oldScale, runningMax, newMax);
                TROWEXPANDMUL(weightedValueSum, weightedValueSum, oldScale);
            }

            // In-place stable exponentiation: probability = exp(score-newMax).
            TROWEXPANDEXPDIF(probabilityLo, probabilityLo, newMax);
            TROWEXPANDEXPDIF(probabilityHi, probabilityHi, newMax);

            // Same PTO #311 carrier as row max. Keep localSum as a zero-copy
            // prefix view; TADD/TFMA consume it through B.SUBVIEW.
            WideBf16RowReductionTile localSumLoWide, localSumHiWide;
            TROWSUM(localSumLoWide, probabilityLo);
            TROWSUM(localSumHiWide, probabilityHi);
            auto localSumLo =
                TREDUCEPREFIXVIEW<Bf16RowValueTile>(localSumLoWide);
            auto localSumHi =
                TREDUCEPREFIXVIEW<Bf16RowValueTile>(localSumHiWide);
            Bf16RowValueTile localSum;
            TADD(localSum, localSumLo, localSumHi);
            Bf16RowValueTile newSum;
            if (kb == 0) {
                TADD(newSum, runningSum, localSum);
            } else {
                // Fuse the BF16 rescale and local contribution. The
                // reduction-prefix addend remains a zero-copy CUBE view.
                TFMA(newSum, runningSum, oldScale, localSum);
            }

            // Partition the BF16 exp result along K into independent 32-value
            // MX groups.  E2M1 data fragments use TASSEMBLY; the four E8M0
            // scale-code columns are compacted separately with TPACK.
            auto blocksLo =
                TPARTVIEW<Bf16ScoreGroupTile, 1,
                          kPScaleColsPerHalf>(probabilityLo);
            auto blocksHi =
                TPARTVIEW<Bf16ScoreGroupTile, 1,
                          kPScaleColsPerHalf>(probabilityHi);
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            auto quantizeGroup = [&]<int Block>(PScaleExponentU8 &scaleCode) {
                auto pBlockView = [&]() {
                    if constexpr (Block < kPScaleColsPerHalf) {
                        return blocksLo[0][Block];
                    } else {
                        return blocksHi[0][Block - kPScaleColsPerHalf];
                    }
                }();

                // Quantize one [32,32] BF16 probability block:
                //   amax  = rowmax(pBlock)             (no TABS: pBlock >= 0)
                //   scale = floor_pow2(amax) / 4       (E2M1 emax = 2)
                //   q     = round_E2M1(pBlock / scale)
                //
                // PTO #311 leaves the per-row amax in the prefix CELL of the
                // wide reduction carrier.  Expose that CELL as a zero-copy
                // view. TMULS consumes the prefix directly, so no compact
                // copy or zero tile is needed to form the BF16 scale.
                WideBf16GroupReductionTile amaxWide;
                TROWMAX(amaxWide, pBlockView);
                auto amaxView = TREDUCEPREFIXVIEW<Bf16RowValueTile>(amaxWide);
                // E2M1 emax=2: scale=floor_pow2(amax)/4.
                Bf16RowValueTile scaleBf16;
                TMULS(scaleBf16, amaxView, kInvFp4PowerMax);
                // RTM is essential here: E8M0 represents powers of two, and
                // MX requires floor(log2), not nearest-exponent rounding.
                // The E8M0 scale retains the M32 layout of scaleBf16.
                Bf16RowValueTile reciprocal;
                auto scaleAsE8M0 = reinterpret_tile<__fp8_e8m0>(scaleCode);
                TCVT<LINX_RDN>(scaleAsE8M0, scaleBf16);
                if constexpr (kBf16RecipFromE8M0) {
                    // Experimental lowp_recip path: round the scale to E8M0,
                    // decode it to BF16, then use the supported BF16 TRECIP.
                    Bf16RowValueTile roundedScaleBf16;
                    TCVT<LINX_RNONE>(roundedScaleBf16, scaleAsE8M0);
                    TRECIP(reciprocal, roundedScaleBf16);
                } else {
                    // E8M0 code e represents 2^(e-127), so its reciprocal
                    // code is 254-e for finite e (0..254).
                    PScaleExponentU8 exponent254;
                    PScaleExponentU8 reciprocalCode;
                    TEXPANDS(exponent254, static_cast<uint8_t>(254));
                    TSUB(reciprocalCode, exponent254, scaleCode);
                    auto reciprocalAsE8M0 =
                        reinterpret_tile<__fp8_e8m0>(reciprocalCode);
                    TCVT<LINX_RNONE>(reciprocal, reciprocalAsE8M0);
                }
                Bf16ScoreGroupTile normalized;
                TROWEXPANDMUL(normalized, pBlockView, reciprocal);
                // Convert the BF16 probabilities directly into the E2M1x2
                // assembly slot.  Avoid an intermediate PBlock and an
                // unnecessary second E2M1x2 -> E2M1x2 TCVT.
                auto pSlot = pFragments[0][Block];
                TCVT(pSlot, normalized);
            };

            // One group-32 scale column is produced by each unrolled call.
            // Pack each consecutive group of four E8M0 bytes into one U32
            // word per row, then store that word into its compact GM column.
            // This named always-inline recursion avoids a separate expansion
            // lambda/call frame, which would force the still-live QK tile to
            // spill through an S64 carrier.
            pack_p_scale_words<0, kPScaleWordCols, PScaleExponentU8,
                               PScaleWordU32, GmPackedPScaleWords>(
                quantizeGroup, packedPScaleScratch);

            // Finish the E2M1 payload assembly.  The packed scale word has the
            // desired byte order [group0, group1, ...], but TPACK leaves U32
            // architectural descriptors.  Reload the same bytes to obtain
            // the compact E8M0 [32,kPScaleCols] descriptor required by PV.
            P p = TASSEMBLY<P>(std::move(pFragments));
            PScale pScale;
            GmPackedPScaleE8M0 packedScaleGm(
                reinterpret_cast<__fp8_e8m0 *>(packedPScaleScratch));
            TLOAD_CUBE(pScale, packedScaleGm);

            // PV matrix stage.  V is an external MXFP4 tensor with its own
            // E8M0 scales.  P uses the just-generated dynamic scales.  V is
            // physically [K,N], hence transpose_b() in pvOptions.
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) +
                      kb * kStoredTk * vD);
            // VScaleMatrix is padded from kPScaleCols columns to
            // kPaddedPScaleCols columns in SharedTReg, while GM V scales are
            // densely stored as [vD, Skv/32].  Select this KV block's first
            // group column and preserve the full-GM row pitch.
            GmVScale gVS(const_cast<__fp8_e8m0 *>(vScalePtr) +
                         kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);
            Bf16PvTile blockPvBf16;
            TMATMUL_MX<3>(blockPvBf16, p, pScale, v, vScale,
                          pvOptions, kGroupM);
            // The earlier weighted-value sum was rescaled if newMax changed.
            // Add this KV block's P * V contribution in the same scale.
            if (kb == 0) weightedValueSum = blockPvBf16;
            else TADD(weightedValueSum, weightedValueSum, blockPvBf16);

            runningMax = newMax;
            runningSum = newSum;
        }

        // Normalize the BF16 weighted-value sum by the BF16 row sum in place.
        TROWEXPANDDIV(weightedValueSum, weightedValueSum, runningSum);
        auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
        TSTORE_CUBE(gOut, weightedValueSum);
    }
}
}  // namespace fa_lowp_ltile
