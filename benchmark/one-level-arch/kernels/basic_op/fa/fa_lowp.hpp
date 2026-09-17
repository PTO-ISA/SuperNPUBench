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
//      matrix.  P and V are then consumed by a second TMATMUL_MX.  PV and the
//      online output accumulator stay FP32 until final normalization, after
//      which O is converted to BF16 and stored.
//
// Current TileOP API notes and compromises:
//
//   * PTO #311 makes a CUBE row-reduction result logically [M,1] but requires
//     a wide physical carrier matching the source columns.  The updated API's
//     TREDUCEPREFIXVIEW exposes its first [M,1] CELL without a copy; mixed
//     TMAX/TADD overloads consume that view directly.  TFMA still has no such
//     overload, so the online-sum update is expressed as TMUL plus TADD.
//
//   * Per-group amax uses the same reduction-prefix view.  TMULS does not yet
//     accept ReductionPrefixView, so a mixed TADD with a compact zero row
//     materializes the prefix before scaling.  This removes the invalid
//     wide-to-compact reduction TCVT while preserving its copy semantics.
namespace fa_lowp {
using namespace pto;

constexpr int kPeNum = 4;       // One cooperative QK/PV matmul uses four PEs.
constexpr int kMxGroup = 32;    // OCP MX scale granularity along matrix K.
constexpr int kPackedFactor = 2;  // E2M1x2 packs two logical FP4 values/byte.

// TODO(TileOP issue filed): the probability scale carrier should be an
// M32-layout E8M0 tile, matching the M32 BF16 row that produces it.  The
// current implementation temporarily routes the scale through a RowMajor
// Scaling tile.  Public TCVT already exposes RMode, but it requires a CUBE_M32
// source to preserve its CUBE_M32 layout, so it cannot express this temporary
// CUBE_M32 <-> RowMajor bridge.  Keep this helper only until the tracked M32
// E8M0 scale path is available.
//
// MX scale generation needs an explicit floor-to-power-of-two conversion:
//   LINX_RDN   -> RTM (round toward minus infinity / floor for positive amax)
//   otherwise  -> RNONE
// It also permits the compact BF16 CUBE row carrier and the padded RowMajor
// Scaling carrier to be connected without pretending they have one C++ tile
// type.  This helper performs a real conversion; it is distinct from the
// compromise wide-to-compact reduction-result TCVTs described above.
template <int RMode, typename Dst, typename Src>
inline void fa_lowp_tcvt(Dst &dst, Src &src) {
    const size_t validCol = src.GetValidCol();
    const size_t validRow = src.GetValidRow();
    asm volatile(
        "BSTART.TEPL 27, %D[SrcT]\n"
        ".if %c[RMode] == 3\nB.DATR %D[DstT], RTM\n"
        ".else\nB.DATR %D[DstT], RNONE\n.endif\n"
        "B.DIM %[VCol], 0, ->lb0\n"
        "B.DIM %[VRow], 0, ->lb1\n"
        "B.DIM zero, %c[Cols], ->lb2\n"
        "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[Size]>\n"
        : [Dst] "=Tr"(dst.data())
        : [SrcT] "i"(type_traits<typename Src::DType>::TypeCode),
          [DstT] "i"(type_traits<typename Dst::DType>::TypeCode),
          [Src] "Tr"(src.data()), [Size] "i"(Dst::TilesizeCode),
          [VCol] "r"(validCol), [VRow] "r"(validRow),
          [Cols] "i"(Dst::Cols), [RMode] "i"(RMode)
        : "memory");
}

// TPARTVIEW is a borrowed range into the parent register, not an independently
// assigned tile.  The subsequent row-reduction, scale decode and data convert
// chain needs a normal assigned CUBE tile.  Multiplication by one materializes
// the selected 32-column range while preserving its BF16 values.  B.SUBVIEW
// selects the range without a GM round trip; TMULS supplies the assigned
// destination tile required by the following operations.
template <typename Out, typename Parent, typename Sub>
inline void materialize_subview(Out &dst,
                                region::SubTileView<Parent, Sub> &src,
                                typename Sub::DType scalar) {
    const uintptr_t base = src.GetRangeBase();
    asm volatile(
        "BSTART.TEPL 34, %D[Type]\n"
        "B.DIM zero, %c[VCol], ->lb0\n"
        "B.DIM zero, %c[VRow], ->lb1\n"
        "B.DIM zero, %c[Cols], ->lb2\n"
        "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[DstSize]>\n"
        "B.SUBVIEW 0, %[Base], 0, %c[SrcSize]\n"
        "B.IOR [%[Scalar]],[]\n"
        : [Dst] "=Tr"(dst.data())
        : [Type] "i"(type_traits<typename Sub::DType>::TypeCode),
          [Src] "Tr"(src.data()), [Base] "r"(base),
          [Scalar] "r"(scalar), [DstSize] "i"(Out::TilesizeCode),
          [SrcSize] "i"(Sub::TilesizeCode), [VCol] "i"(Sub::ValidCol),
          [VRow] "i"(Sub::ValidRow), [Cols] "i"(Sub::Cols)
        : "memory");
}

template <int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
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

    // Per-PE QK/softmax tiles.  Score is the BF16 fixpipe destination.
    // Reduce/BlockReduce deliberately retain the source's physical columns
    // with ValidCol=1 to satisfy PTO #311.  Row is the compact descriptor
    // required by TMAX, TFMA and TROWEXPAND*.
    using Score = CubeTileM32<__bf16, kPeM, kTk>;
    using Reduce = VecTileM32<__bf16, kPeM, kTk, kPeM, 1>;
    using Row = VecTileM32<__bf16, kPeM, 1, kPeM, 1>;
    using ScoreBlock = CubeTileM32<__bf16, kPeM, kMxGroup>;
    using BlockReduce = VecTileM32<__bf16, kPeM, kMxGroup, kPeM, 1>;
    // P is quantized group-by-group.  PBlock contains 32 logical E2M1 values
    // per row; PScaleFragment contains the corresponding one valid scale.
    //
    // TODO(TileOP issue filed): PScaleFragment/PScale should use an M32 E8M0
    // layout.  RowMajor Scaling below is only the current API workaround and
    // must not be treated as the intended scale layout.  It is physically
    // padded to four columns (the 128 B minimum), while only the first column
    // is valid.  TASSEMBLY concatenates these temporary physical fragments;
    // PScale exposes exactly kTk/32 valid scale columns to TMATMUL_MX.
    using PBlock = CubeTileM32<__fp4_e2m1x2, kPeM, kMxGroup>;
    using P = CubeTileM32<__fp4_e2m1x2, kPeM, kTk>;
    using PScaleFragment = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, 4, BLayout::RowMajor, kPeM, 1>;
    using PScale = Tile<Location::Scaling, __fp8_e8m0,
        kPeM, 4 * kPScaleCols, BLayout::RowMajor, kPeM, kPScaleCols>;
    using PV = CubeAccumulatorM32<float, kPeM, vD>;
    using OCast = CubeAccumulatorM32<__bf16, kPeM, vD>;
    using FloatRow = VecTileM32<float, kPeM, 1, kPeM, 1>;

    using QScaleIter = global_iterator<GmQScale, QScaleMatrix>;
    using KScaleIter = global_iterator<GmKScale, KScaleMatrix>;
    using VScaleIter = global_iterator<GmVScale, VScaleMatrix>;
    using OIter = global_iterator<GmO, OCast>;
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
                  PScaleFragment::LogicalTileBytes * kPScaleCols);

    // E2M1 has maximum exponent emax=2.  MX scaling is based on the exponent
    // bound 2^emax=4, not on the maximum finite E2M1 value 6:
    //   scale = 2^floor(log2(amax)) / 4.
    // Multiplying by 1/4 and converting to E8M0 with RTM implements this.
    const __bf16 qkScale = static_cast<__bf16>(
        1.0f / sqrt(static_cast<float>(scaleD)));
    const __bf16 kInvFp4PowerMax = static_cast<__bf16>(0.25f);
    constexpr auto qkOptions = fixp::bf16();
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

#pragma clang loop unroll(full)
    for (int qb = 0; qb < kQBlocks; ++qb) {
        // Online-softmax state for this PE's 32 query rows.  out is FP32 to
        // protect the PV accumulation; max/sum stay BF16 as requested.
        PV out;
        Row runningMax, runningSum;
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
            Score probability;
            TMATMUL_MX<3>(probability, q, qScale, k, kScale, qkOptions);
            TMULS(probability, probability, qkScale);

            // Row max for numerically stable online softmax.  PTO #311 keeps
            // the [32,1] result in the prefix CELL of a wide carrier; expose
            // that CELL as a zero-copy reduction-prefix view.
            Reduce localMaxWide;
            TROWMAX(localMaxWide, probability);
            auto localMax = TREDUCEPREFIXVIEW<Row>(localMaxWide);
            // Online max update.  If the max changes, oldScale computes
            // exp(runningMax-newMax) and rescales the already accumulated PV
            // contribution.  out is FP32, so the BF16 row scale is widened
            // before broadcasting over the FP32 accumulator.
            Row newMax, oldScale;
            TMAX(newMax, runningMax, localMax);
            if (kb != 0) {
                TROWEXPANDEXPDIF(oldScale, runningMax, newMax);
                FloatRow oldScaleF;
                TCVT(oldScaleF, oldScale);
                TROWEXPANDMUL(out, out, oldScaleF);
            }

            // In-place stable exponentiation: probability = exp(score-newMax).
            TROWEXPANDEXPDIF(probability, probability, newMax);

            // Same PTO #311 carrier as row max.  Keep localSum as a zero-copy
            // prefix view and let TADD consume it through B.SUBVIEW.
            Reduce localSumWide;
            TROWSUM(localSumWide, probability);
            auto localSum = TREDUCEPREFIXVIEW<Row>(localSumWide);
            Row newSum;
            if (kb == 0) {
                TADD(newSum, runningSum, localSum);
            } else {
                // No TFMA reduction-prefix overload yet: split the update so
                // TADD can consume localSum without a compact TCVT copy.
                // BF16 multiplication and addition round separately here.
                Row scaledSum;
                TMUL(scaledSum, runningSum, oldScale);
                TADD(newSum, scaledSum, localSum);
            }

            // Partition the BF16 exp result along K into independent 32-value
            // MX groups.  Data fragments and scale fragments are collected in
            // separate assembly sessions so PV receives two complete tiles.
            auto blocks = TPARTVIEW<ScoreBlock, 1, kPScaleCols>(probability);
            TileArray<PBlock, 1, kPScaleCols> pFragments;
            TileArray<PScaleFragment, 1, kPScaleCols> pScaleFragments;
            __bf16 one = static_cast<__bf16>(1.0f);
            asm volatile("" : "+r"(one) : : "memory");

#pragma clang loop unroll(full)
            for (int block = 0; block < kPScaleCols; ++block) {
                auto view = blocks[0][block];
                ScoreBlock pBlock;
                materialize_subview(pBlock, view, one);

                // Quantize one [32,32] BF16 probability block:
                //   amax  = rowmax(pBlock)             (no TABS: pBlock >= 0)
                //   scale = floor_pow2(amax) / 4       (E2M1 emax = 2)
                //   q     = round_E2M1(pBlock / scale)
                //
                // PTO #311 leaves the per-row amax in the prefix CELL of the
                // wide reduction carrier.  Expose that CELL as a zero-copy
                // view.  TMULS cannot consume ReductionPrefixView directly,
                // so TADD with zero materializes a normal compact Row without
                // using the invalid wide-to-compact CUBE TCVT.
                BlockReduce amaxWide;
                TROWMAX(amaxWide, pBlock);
                auto amaxView = TREDUCEPREFIXVIEW<Row>(amaxWide);
                // E2M1 emax=2: scale=floor_pow2(amax)/4.
                Row zero;
                TEXPANDS(zero, 0.0f);
                Row scaleBf16;
                TADD(scaleBf16, zero, amaxView);
                TMULS(scaleBf16, scaleBf16, kInvFp4PowerMax);
                // RTM is essential here: E8M0 represents powers of two, and
                // MX requires floor(log2), not nearest-exponent rounding.
                // TODO(TileOP issue filed): replace this bridge with public
                // TCVT<LINX_RDN> once scale is represented as M32 E8M0.
                PScaleFragment scale;
                fa_lowp_tcvt<LINX_RDN>(scale, scaleBf16);

                // Decode the quantized E8M0 scale back to BF16, form its
                // reciprocal, normalize the block, and finally encode E2M1x2.
                // This reverse bridge is part of the same temporary RowMajor
                // scale workaround; the intended scale carrier is M32 E8M0.
                Row decodedScale;
                fa_lowp_tcvt<LINX_RNONE>(decodedScale, scale);
                Row reciprocal;
                TRECIP(reciprocal, decodedScale);
                ScoreBlock normalized;
                TROWEXPANDMUL(normalized, pBlock, reciprocal);
                PBlock quantized;
                TCVT(quantized, normalized);

                // These TCVTs write into TileArray assembly slots.  They are
                // staging copies into the data/scale assembly carriers, not a
                // change to the MX scale definition.
                auto pSlot = pFragments[0][block];
                auto sSlot = pScaleFragments[0][block];
                TCVT(pSlot, quantized);
                TCVT(sSlot, scale);
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
            PV current;
            TMATMUL_MX<3>(current, p, pScale, v, vScale,
                          pvOptions, kGroupM);
            // out was already rescaled above if newMax changed, so the current
            // KV block contribution can now be accumulated directly.
            if (kb == 0) out = current;
            else TADD(out, out, current);

            runningMax = newMax;
            runningSum = newSum;
        }

        // Complete softmax normalization after all KV blocks.  PV is FP32, so
        // widen the BF16 running sum for the broadcast divide, convert the
        // normalized result to BF16, and store this PE's 32 output rows.
        FloatRow sumF;
        TCVT(sumF, runningSum);
        TROWEXPANDDIV(out, out, sumF);
        OCast result;
        TCVT(result, out);
        auto gOut = outIter(qb * kPeNum + static_cast<int>(tid), 0);
        TSTORE_CUBE(gOut, result);
    }
}
}  // namespace fa_lowp
