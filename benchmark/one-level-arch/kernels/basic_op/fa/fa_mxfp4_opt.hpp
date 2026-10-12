#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <utility>

// PMU-optimized MXFP4 FlashAttention
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
//      TPACK compacts four E8M0 scale bytes into the complete P-scale matrix.
//      P and V are then consumed by a second TMATMUL_MX.  Each PV result, the
//      online weighted-value sum, and final normalization all remain BF16;
//      the normalized BF16 Tile is stored directly as O.
//
// Scheduling changes relative to fa_lowp.hpp:
//
//   * V and V-scale TLOADs are issued as soon as QK releases the K shared
//     operands.  Their memory latency can then overlap the online-softmax and
//     P-quantization vector pipeline instead of sitting directly in front of
//     the PV TMATMUL_MX critical path.
//
//   * The constant E8M0 reciprocal bias (254) is materialized once per Q
//     block and shared by every group-32 quantizer in the fully-unrolled KV
//     loop.  The reference kernel materializes the same 128-B vector four
//     times per KV block.
//
// Current TileOP API notes and compromises:
//
//   * PTO #311 makes a CUBE row-reduction result logically [M,1] but requires
//     a wide physical carrier matching the source columns.  The updated API's
//     TREDUCEPREFIXVIEW exposes its first [M,1] CELL without a copy; mixed
//     TMAX/TADD/TFMA overloads consume that view directly.
//
//   * Per-group amax uses the same reduction-prefix view. TMULS consumes its
//     prefix directly, avoiding a wide-to-compact reduction TCVT or copy.
namespace fa_mxfp4_opt {
using namespace pto;

constexpr int kPeNum = 4;       // One cooperative QK/PV matmul uses four PEs.
constexpr int kMxGroup = 32;    // OCP MX scale granularity along matrix K.
constexpr int kPackedFactor = 2;  // E2M1x2 packs two logical FP4 values/byte.

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD, bool kBf16RecipFromE8M0 = false>
void flash_attention_mxfp4_opt_impl(
    __bf16 *outPtr, const __fp4_e2m1x2 *qPtr,
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

    // This first implementation intentionally fixes the cooperative shape to
    // 128 rows: four PEs each own one 32-row CUBE_M32 result shard.
    static_assert(kGroupM == 128,
                  "fa_mxfp4_opt requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    static_assert((kPScaleCols & (kPScaleCols - 1)) == 0,
                  "P scale packing requires a power-of-two block count");

    // Packed GM data layouts.  The stored K dimension is divided by two
    // because every E2M1x2 byte contains two adjacent logical K values.
    // V is packed along its reduction/K dimension as [Skv/2, vD].
    using QSlice = global_tensor<__fp4_e2m1x2,
                                 RowMajor<kGroupM, kStoredQD>>;
    using KSlice = global_tensor<__fp4_e2m1x2,
                                 RowMajor<kTk, kStoredQD>>;
    using VSlice = global_tensor<__fp4_e2m1x2,
                                 RowMajor<kStoredTk, vD>>;
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
    using KMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, qD>;
    using VMatrix = SharedMatrixRight<__fp4_e2m1x2, kTk, vD>;
    using QTile = SharedTile<QMatrix>;
    using KTile = SharedTile<KMatrix>;
    using VTile = SharedTile<VMatrix>;

    // Shared scale tiles are padded to the minimum legal physical matrix
    // carrier while ValidCol/ValidRow describe only the real group-32 scales.
    // TMATMUL_MX validates and consumes the valid shapes, not the padding.
    using QScaleMatrix = SharedMatrixLeft<
        __fp8_e8m0, 128, kPaddedQScaleCols, 128, kQScaleCols>;
    using KScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, kTk, kPaddedQScaleCols, kTk, kQScaleCols>;
    using VScaleMatrix = SharedMatrixRight<
        __fp8_e8m0, vD, kPaddedPScaleCols, vD, kPScaleCols>;
    using QScaleTile = SharedTile<QScaleMatrix>;
    using KScaleTile = SharedTile<KScaleMatrix>;
    using VScaleTile = SharedTile<VScaleMatrix>;

    // Per-PE QK/softmax tiles.  A reduction carrier keeps the source's
    // physical column span but has only one valid value per row (PTO #311).
    // A row-value tile is the compact, one-CELL form of those row scalars:
    // one valid column, two physical BF16 columns in CUBE_M32.
    using Bf16ScoreTile = CubeTileM32<__bf16, kPeM, kTk>;
    using WideBf16RowReductionTile =
        VecTileM32<__bf16, kPeM, kTk, kPeM, 1>;
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
    // Four 32x1 raw scale-code columns are packed byte-wise into one 32x1 U32
    // CUBE_M32 cell.  The same 128 B payload is then consumed as a compact
    // 32x4 E8M0 scale tile by TMATMUL_MX.  TASSEMBLY cannot be used here: it
    // would concatenate four padded 128 B fragments and place their live
    // columns at physical offsets 0/4/8/12 instead of compact columns 0..3.
    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    using PScaleExponentU8 = Tile<Location::Scaling, uint8_t,
        kPeM, 4, BLayout::CubeM32, kPeM, 1>;
    using PScaleWordU32 = Tile<Location::Scaling, uint32_t,
        kPeM, 1, BLayout::CubeM32, kPeM, 1>;
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, 4, BLayout::CubeM32, kPeM, kPScaleCols>;
    using GmPackedPScaleWords =
        global_tensor<uint32_t, RowMajor<kPeM, 1>>;
    using GmPackedPScaleE8M0 =
        global_tensor<__fp8_e8m0, RowMajor<kPeM, kPScaleCols>>;
    using Bf16WeightedValueTile = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using Bf16PvTile = Bf16WeightedValueTile;

    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16WeightedValueTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KScaleIter kScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    OIter outIter(outPtr);

    // Shared Q/K and their scales coexist during QK; V and its scale coexist
    // during PV.  Each phase must fit the 256-KiB cooperative SharedTReg pool.
    static_assert(QMatrix::LogicalTileBytes + KMatrix::LogicalTileBytes +
                      QScaleMatrix::LogicalTileBytes +
                      KScaleMatrix::LogicalTileBytes <= 256 * 1024);
    static_assert(VMatrix::LogicalTileBytes +
                      VScaleMatrix::LogicalTileBytes <= 256 * 1024);
    static_assert(P::LogicalTileBytes ==
                  PBlock::LogicalTileBytes * kPScaleCols);
    static_assert(kPScaleCols == 4,
                  "fa_mxfp4_opt currently packs exactly four group-32 scales");
    static_assert(PScale::LogicalTileBytes == PScaleWordU32::LogicalTileBytes,
                  "packed U32 words and E8M0 scale tile must share one cell");

    // TPACK is specified to produce a U32 Tile.  A C++ bit_cast/reinterpret
    // changes only the source-level type and does not retag the architectural
    // Tile-register descriptor, so TMATMUL_MX cannot consume that register as
    // E8M0 directly.  Use one private 128 B stack slot as a storage-preserving
    // U32 -> E8M0 retag bridge until TileOP provides a cross-element-width
    // storage reinterpret operation.
    alignas(128) uint32_t packedPScaleScratch[kPeM];

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

        // Keep one reciprocal-bias Tile alive across the fully-unrolled KV
        // loop. The compiler uses cheap hand-to-hand TMOVs when necessary;
        // this still costs less than rematerializing the vector per group.
        PScaleExponentU8 exponent254;
        if constexpr (!kBf16RecipFromE8M0) {
            TEXPANDS(exponent254, static_cast<uint8_t>(254));
        }

#pragma clang loop unroll(full)
        for (int kb = 0; kb < kKVBlocks; ++kb) {
            // Load the next [kTk,qD] K block and its group-32 E8M0 scales.
            KTile k;
            KScaleTile kScale;
            KSlice gK(const_cast<__fp4_e2m1x2 *>(kPtr) +
                      kb * kTk * kStoredQD);
            auto gKS = kScaleIter(kb, 0);
            TLOAD<KMatrix, 1>(k, gK);
            TLOAD<KScaleMatrix, 1>(kScale, gKS);

            // QK matrix stage.  TMATMUL_MX dequantizes Q/K using both scale
            // matrices.  fixp::bf16() makes probability BF16 at the matrix
            // output; there is intentionally no FP4 conversion here.
            Bf16ScoreTile probability;
            TMATMUL_MX<3>(probability, q, qScale, k, kScale, qkOptions);

            // QK has consumed S2/S3 (K/K-scale), so reuse those shared
            // handles immediately for V/V-scale.  Keeping these loads ahead
            // of the long BF16 softmax/quantization chain exposes their
            // latency to TLSU/Vector overlap; PV still depends on them in the
            // usual way and therefore needs no explicit barrier here.
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) +
                      kb * kStoredTk * vD);
            GmVScale gVS(const_cast<__fp8_e8m0 *>(vScalePtr) +
                         kb * kPScaleCols);
            TLOAD<VMatrix, 1>(v, gV);
            TLOAD<VScaleMatrix, 1>(vScale, gVS);

            TMULS(probability, probability, qkScale);

            // Row max for numerically stable online softmax.  PTO #311 keeps
            // the [32,1] result in the prefix CELL of a wide carrier; expose
            // that CELL as a zero-copy reduction-prefix view.
            WideBf16RowReductionTile localMaxWide;
            TROWMAX(localMaxWide, probability);
            // subview提取出dst tile中valid 一列
            auto localMax = TREDUCEPREFIXVIEW<Bf16RowValueTile>(localMaxWide);
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
            TROWEXPANDEXPDIF(probability, probability, newMax);

            // Same PTO #311 carrier as row max. Keep localSum as a zero-copy
            // prefix view; TADD/TFMA consume it through B.SUBVIEW.
            WideBf16RowReductionTile localSumWide;
            TROWSUM(localSumWide, probability);
            // subview提取出dst tile中valid 一列
            auto localSum = TREDUCEPREFIXVIEW<Bf16RowValueTile>(localSumWide);
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
            auto blocks = TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(probability);
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            auto quantizeGroup = [&]<int Block>(PScaleExponentU8 &scaleCode) {
                auto pBlockView = blocks[0][Block];

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
                    PScaleExponentU8 reciprocalCode;
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
            // Keep the raw E8M0 exponent bytes in U8 carriers so TCVT can
            // widen them value-preservingly before TPACK.
            PScaleExponentU8 scaleCode0, scaleCode1, scaleCode2, scaleCode3;
            quantizeGroup.template operator()<0>(scaleCode0);
            quantizeGroup.template operator()<1>(scaleCode1);
            quantizeGroup.template operator()<2>(scaleCode2);
            quantizeGroup.template operator()<3>(scaleCode3);

            PScaleWordU32 scaleWord0, scaleWord1, scaleWord2, scaleWord3;
            TCVT(scaleWord0, scaleCode0);
            TCVT(scaleWord1, scaleCode1);
            TCVT(scaleWord2, scaleCode2);
            TCVT(scaleWord3, scaleCode3);

            PScaleWordU32 scalePair01, scalePair23, packedScaleWords;
            TPACK(scalePair01, scaleWord0, scaleWord1, 0x00000101);
            TPACK(scalePair23, scaleWord2, scaleWord3, 0x00000101);
            TPACK(packedScaleWords, scalePair01, scalePair23, 0x00000202);

            // Finish the E2M1 payload assembly.  The packed scale word has the
            // desired byte order [group0, group1, group2, group3], but TPACK
            // leaves a U32 architectural descriptor.  Store/reload the same
            // 128 bytes to obtain the E8M0 [32,4] descriptor required by PV.
            P p = TASSEMBLY<P>(std::move(pFragments));
            GmPackedPScaleWords packedWordsGm(packedPScaleScratch);
            TSTORE_CUBE(packedWordsGm, packedScaleWords);
            PScale pScale;
            GmPackedPScaleE8M0 packedScaleGm(
                reinterpret_cast<__fp8_e8m0 *>(packedPScaleScratch));
            TLOAD_CUBE(pScale, packedScaleGm);

            // PV matrix stage.  V is an external MXFP4 tensor with its own
            // E8M0 scales.  P uses the just-generated dynamic scales.  V is
            // physically [K,N], hence transpose_b() in pvOptions.
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
}  // namespace fa_mxfp4_opt
