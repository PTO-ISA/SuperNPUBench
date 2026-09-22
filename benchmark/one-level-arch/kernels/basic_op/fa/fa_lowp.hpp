#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <utility>

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
//     TMAX/TADD overloads consume that view directly.  TFMA still has no such
//     overload, so the online-sum update is expressed as TMUL plus TADD.
//
//   * Per-group amax uses the same reduction-prefix view. TMULS consumes its
//     prefix directly, avoiding a wide-to-compact reduction TCVT or copy.
namespace fa_lowp {
using namespace pto;

constexpr int kPeNum = 4;       // One cooperative QK/PV matmul uses four PEs.
constexpr int kMxGroup = 32;    // OCP MX scale granularity along matrix K.
constexpr int kPackedFactor = 2;  // E2M1x2 packs two logical FP4 values/byte.

// TPARTVIEW is a borrowed range into the parent tile. Materialize the
// selected range for TROWEXPANDMUL, which currently requires a tile source.
template <typename Out, typename Parent, typename Sub>
inline void materialize_subview(Out &dst,
                                region::SubTileView<Parent, Sub> &src,
                                typename Sub::DType scalar) {
    TMULS(dst, src, scalar);
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
                  "fa_lowp requires a 128-row cooperative Q tile");
    static_assert(Sq % kGroupM == 0 && Skv % kTk == 0);
    static_assert(qD % kMxGroup == 0 && kTk % kMxGroup == 0);
    static_assert((kPScaleCols & (kPScaleCols - 1)) == 0,
                  "P scale assembly requires a power-of-two block count");

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
    // Q scale [Sq,qD/32], K scale [Skv,qD/32], V scale [Skv/32,vD].
    using GmQScale = global_tensor<__fp8_e8m0,
                                   RowMajor<Sq, kQScaleCols>>;
    using GmKScale = global_tensor<__fp8_e8m0,
                                   RowMajor<Skv, kQScaleCols>>;
    using GmVScale = global_tensor<__fp8_e8m0,
                                   RowMajor<Skv / kMxGroup, vD>>;
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
        __fp8_e8m0, kPaddedPScaleCols, vD, kPScaleCols, vD>;
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
    // values per row; PScaleFragmentFp8E8M0 contains one valid E8M0 scale.
    // __fp4_e2m1x2 already packs two logical 4-bit values in one byte, so the
    // Tile column count remains the logical column count and must not be
    // doubled to account for the carrier representation.
    //
    // The scale uses the same M32 layout as its BF16 source.  It is physically
    // padded to four columns (the 128 B minimum), while only the first column
    // is valid.  TASSEMBLY concatenates these physical fragments;
    // PScale exposes exactly kTk/32 valid scale columns to TMATMUL_MX.
    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    using PScaleFragmentFp8E8M0 = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, 4, BLayout::CubeM32, kPeM, 1>;
    using PScaleExponentU8 = Tile<Location::Scaling, uint8_t,
        kPeM, 4, BLayout::CubeM32, kPeM, 1>;
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, 4 * kPScaleCols, BLayout::CubeM32, kPeM, kPScaleCols>;
    using Bf16WeightedValueTile = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using Bf16PvTile = Bf16WeightedValueTile;

    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using VScaleIter = global_iterator<GmVScale, VScaleMatrix>;
    using OIter = global_iterator<GmO, Bf16WeightedValueTile>;
    QScaleIter qScaleIter(const_cast<__fp8_e8m0 *>(qScalePtr));
    KScaleIter kScaleIter(const_cast<__fp8_e8m0 *>(kScalePtr));
    VScaleIter vScaleIter(const_cast<__fp8_e8m0 *>(vScalePtr));
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
    static_assert(PScale::LogicalTileBytes ==
                  PScaleFragmentFp8E8M0::LogicalTileBytes * kPScaleCols);

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
            // MX groups.  Data fragments and scale fragments are collected in
            // separate assembly sessions so PV receives two complete tiles.
            auto blocks = TPARTVIEW<Bf16ScoreGroupTile, 1, kPScaleCols>(probability);
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            TileArray<PScaleFragmentFp8E8M0, 1, kPScaleCols> pScaleFragments;
#pragma clang loop unroll(full)
            // online quantization
            for (int block = 0; block < kPScaleCols; ++block) {
                auto view = blocks[0][block];
                Bf16ScoreGroupTile pBlock;
                materialize_subview(pBlock, view, static_cast<__bf16>(1.0f));

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
                TROWMAX(amaxWide, pBlock);
                auto amaxView = TREDUCEPREFIXVIEW<Bf16RowValueTile>(amaxWide);
                // E2M1 emax=2: scale=floor_pow2(amax)/4.
                Bf16RowValueTile scaleBf16;
                TMULS(scaleBf16, amaxView, kInvFp4PowerMax);
                // RTM is essential here: E8M0 represents powers of two, and
                // MX requires floor(log2), not nearest-exponent rounding.
                // The E8M0 scale retains the M32 layout of scaleBf16.
                Bf16RowValueTile reciprocal;
                if constexpr (kBf16RecipFromE8M0) {
                    // Experimental lowp_recip path: round the scale to E8M0,
                    // decode it to BF16, then use the supported BF16 TRECIP.
                    PScaleFragmentFp8E8M0 scaleE8M0;
                    Bf16RowValueTile roundedScaleBf16;
                    TCVT<LINX_RDN>(scaleE8M0, scaleBf16);
                    TCVT<LINX_RNONE>(roundedScaleBf16, scaleE8M0);
                    TRECIP(reciprocal, roundedScaleBf16);
                } else {
                    PScaleExponentU8 scaleCode;
                    auto scaleAsE8M0 = reinterpret_tile<__fp8_e8m0>(scaleCode);
                    TCVT<LINX_RDN>(scaleAsE8M0, scaleBf16);

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
                TROWEXPANDMUL(normalized, pBlock, reciprocal);
                // Convert the BF16 probabilities directly into the E2M1x2
                // assembly slot.  Avoid an intermediate PBlock and an
                // unnecessary second E2M1x2 -> E2M1x2 TCVT.
                auto pSlot = pFragments[0][block];
                auto sSlot = pScaleFragments[0][block];
                TCVT(pSlot, normalized);
                // E8M0 cannot be a TCVT source under the ISA hardware profile.
                // Re-encode the BF16 scale directly into its assembly slot
                // with RTM, matching the standalone scale used above.

                // 增加tpack hif4
                TCVT<LINX_RDN>(sSlot, scaleBf16);
            }

            // Finish both assembly sessions.  p has logical shape [32,kTk];
            // pScale has logical valid shape [32,kTk/32].
            P p = TASSEMBLY<P>(std::move(pFragments));
            PScale pScale = TASSEMBLY<PScale>(std::move(pScaleFragments));

            // PV matrix stage.  V is an external MXFP4 tensor with its own
            // E8M0 scales.  P uses the just-generated dynamic scales.  V is
            // physically [K,N], hence transpose_b() in pvOptions.
            VTile v;
            VScaleTile vScale;
            VSlice gV(const_cast<__fp4_e2m1x2 *>(vPtr) +
                      kb * kStoredTk * vD);
            auto gVS = vScaleIter(kb, 0);
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
}  // namespace fa_lowp
