#pragma once

// matmul_quantize: fused 4-PE matmul -> block quantize -> (data + scale).
//
// Phase 1 (cooperative 4-PE matmul): delegates to matmul_shared, which
// accumulates the FP32 result C = A * B into a scratch GM buffer
// (SharedMatrixLeft A x SharedMatrixRight B, internal-acc K-chain,
// TSTORE_CUBE). Every PE contributes its CUBE row slice of each output
// tile, so the full [gM, gN] FP32 C is materialised in c_scratch.
//
// Phase 1 -> Phase 2 hand-off: NO barrier. matmul_shared partitions C by a
// strided CUBE row mapping -- PE `tid` writes rows (i*4+tid)*kPeM for each
// Mb block i (see `gIterC(i*kPeNum+tid, j)` in matmul_shared.hpp). Phase 2
// quantises the *same* strided stripes, so PE tid's Phase-2 TLOADs read
// only the rows PE tid's Phase-1 TSTORE_CUBE wrote. That eliminates the
// cross-PE dependency. On a functional core (gfrun) program order on the
// same PE makes the Phase-1 store visible to the Phase-2 load, so no
// software barrier or fence is needed. (A timing model with a store
// buffer may still want a fence; that is out of scope for the functional
// verification target here.)
//
// This coupling requires kPeM==32 (the 32-row quantise tile) and so
// kGroupM>64, i.e. tM in {128, 256, ...} (tM<=128 -> kGroupM=tM>64).
// kGroupM/kPeM/kPeNum are mirrored verbatim from matmul_shared.hpp; if
// that kernel's row partition changes, the static_assert below fires.
//
// Phase 2 (SPMD 4-PE quantize): per 32-wide MX block:
//   TLOAD fp32 C [32,32] -> TCVT fp32->bf16 (enter the bf16 domain so the
//   TROWMAX source stays <= 2048B, the #585 row-reduce budget) ->
//   mxquant::quantize_block (TABS / TROWMAX / E8M0-OCP RTM convert /
//   reciprocal code / TROWEXPANDMUL) -> scaled bf16 [32,32] + E8M0 scale
//   code [32,1] -> TCVT scaled bf16 -> OutT -> TSTORE data + TSTORE scale.
//
// Assemble path (kUseAssemble=true): two adjacent 32-wide blocks' scaled
// bf16 halves are packed into one [32,64] bf16 parent via
// TASSEMBLY<Tile>(TileArray) -- the mxquant.hpp `#if 0` pattern, enabled
// here (it compiles and lowers to B.ASSEMBLE, confirmed by a probe). The
// packed parent is then TCVT'd to OutT [32,64] and stored in one TSTORE.
// gfrun (functional) models B.ASSEMBLE -- the assemble path passes gfrun
// with R2=0 (verified); gfsim (timing) does not model it, so the assemble
// path is not gfsim-runnable. The strided path (kUseAssemble=false) stores
// each 32-wide half separately and is both gfrun- and gfsim-runnable.
//
// Output format (default OutT=__fp8_e4m3): E4M3 data + per-row E8M0 scale,
// group 32 -- the same MX structure as mxquant.hpp. This is an
// implementable HiF4-style block quantize; a true E6M2 + U32-scale HiF4 is
// unimplementable in the current TileOP (no E6M2 encode intrinsic; U32
// scale rejected as MX input) and is left as a documented stretch (bottom).
//
// Template params:
//   dtype   - A/B input element type (float / __bf16 / __half). C is FP32.
//   gM/gN/gK - global M/N/K. gN must be a multiple of 32 (one MX block) and,
//             for kUseAssemble, of 64 (a block pair). gM must divide
//             kGroupM (see matmul_shared).
//   tM/tN/tK - tile sizes forwarded to matmul_shared (cooperative TMATMUL).
//   kUseAssemble - true:  TASSEMBLY pack two halves (compile/disasm only).
//             false: strided half stores (gfrun/gfsim runnable).
//   OutT    - quantized data dtype. Default __fp8_e4m3 (mxquant format,
//             proven bf16->E4M3 TCVT). __fp4_e2m1x2 selects the FP4 variant.

#include "basic_op/matmul/matmul_shared.hpp"
#include "basic_op/mxquant/mxquant.hpp"
#include <common/pto_tileop.hpp>
#include <cstdint>

namespace matmul_quantize_detail {

using namespace pto;

constexpr int kPeNum = 4;
constexpr int kQuantRows = 32;   // mxquant::quantize_block is fixed to 32 rows
constexpr int kBlock = 32;        // MX group width

// Phase-2 source tile: FP32 C block [32,32] (strided read from c_scratch).
using CBlockTile = Tile<Location::Vec, float, kQuantRows, kBlock, BLayout::RowMajor>;
// bf16 quantize-domain tiles (reused verbatim from mxquant so
// quantize_block's exact operand types match).
using BfBlockTile = mxquant::BlockTile;   // Tile<Vec,__bf16,32,32,RowMajor>
using BfFullTile = mxquant::InputTile;    // Tile<Vec,__bf16,32,64,RowMajor>

// Quantized output tiles. OutT defaults to __fp8_e4m3 (byte-per-element);
// __fp4_e2m1x2 is the packed FP4 variant (gfrun packs 2 elements/byte on
// TSTORE, matching dynamic_mx_quant_tail_ocp_fp4's convention).
template <typename OutT>
using QuantHalfTile =
    Tile<Location::Vec, OutT, kQuantRows, kBlock, BLayout::RowMajor>;
template <typename OutT>
using QuantTile =
    Tile<Location::Vec, OutT, kQuantRows, 2 * kBlock, BLayout::RowMajor>;

}  // namespace matmul_quantize_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK,
          bool kUseAssemble = true, typename OutT = __fp8_e4m3>
void matmul_quantize(OutT *data_ptr, uint8_t *scale_ptr, float *c_scratch,
                     dtype *a_ptr, dtype *b_ptr) {
    using namespace matmul_quantize_detail;
    namespace mq = mxquant;

    // Mirror matmul_shared.hpp's cooperative row partition so Phase 2 reads
    // only rows its own PE wrote in Phase 1 (no cross-PE dependency, no
    // barrier -- see file header).
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
    constexpr int kValidRowM = (gM < kTileRows) ? gM : kTileRows;
    constexpr int kPeM = kValidRowM <= 64 ? 16 : 32;
    constexpr int Mb = (gM + kGroupM - 1) / kGroupM;
    static_assert(kPeM == kQuantRows,
                  "Phase-2 32-row quantize tiles require kPeM==32, i.e. "
                  "tM>=128 (kGroupM>64). Smaller tM would need a 16-row "
                  "quantize, which mxquant::quantize_block does not provide.");
    static_assert(gM % kGroupM == 0 || gM < kGroupM,
                  "gM must divide matmul_shared's kGroupM or be smaller than it");
    static_assert(gM % kPeM == 0 || gM < kGroupM,
                  "gM must be a multiple of kPeM, or a partial tile (gM < kGroupM)");
    static_assert(gN % kBlock == 0,
                  "gN must be a multiple of the 32-wide MX block");
    if constexpr (kUseAssemble)
        static_assert(gN % (2 * kBlock) == 0,
                      "assemble path pairs two 32-wide blocks; gN must be a "
                      "multiple of 64");

    constexpr int numKb = gN / kBlock;

    // ---- Phase 1: cooperative 4-PE matmul -> FP32 C in c_scratch ------
    matmul_shared<dtype, gM, gN, gK, tM, tN, tK>(c_scratch, a_ptr, b_ptr);

    // ---- Phase 2: SPMD quantize, strided row partition (matches Phase 1) -
    const uint32_t tid = get_thread_idx();

    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using gmData = global_tensor<OutT, RowMajor<gM, gN>>;
    using gmScale = global_tensor<uint8_t, RowMajor<gM, numKb>>;

    #pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
        // Same row mapping as matmul_shared's gIterC(i*kPeNum+tid, j).
        const int row0 = (i * kPeNum + static_cast<int>(tid)) * kPeM;

        if constexpr (gM < kGroupM) {
            if (row0 >= gM) continue;
        }

        if constexpr (kUseAssemble) {
            // Pair two 32-wide blocks into one [32,64] OutT tile via
            // TASSEMBLY (lowers to B.ASSEMBLE). gfsim (timing) does not
            // model B.ASSEMBLE; gfrun (functional) does.
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < numKb; kb += 2) {
                // --- block A (kb) ---
                global_iterator<gmC, CBlockTile> itCA(
                    c_scratch + row0 * gN + kb * kBlock);
                auto gCA = itCA(0, 0);
                CBlockTile cA;
                TLOAD(cA, gCA);

                BfBlockTile bfA;
                TCVT(bfA, cA);                  // fp32 -> bf16

                BfBlockTile scaledA;
                mq::ScaleCodeTile scaleA;
                mq::quantize_block(scaledA, scaleA, bfA);

                // --- block B (kb+1) ---
                global_iterator<gmC, CBlockTile> itCB(
                    c_scratch + row0 * gN + (kb + 1) * kBlock);
                auto gCB = itCB(0, 0);
                CBlockTile cB;
                TLOAD(cB, gCB);

                BfBlockTile bfB;
                TCVT(bfB, cB);

                BfBlockTile scaledB;
                mq::ScaleCodeTile scaleB;
                mq::quantize_block(scaledB, scaleB, bfB);

                // --- assemble two bf16 halves -> one [32,64] bf16 parent ---
                TileArray<BfBlockTile, 1, 2> scaledParts;
                TCVT(scaledParts[0][0], scaledA);   // same-dtype slot copy
                TCVT(scaledParts[0][1], scaledB);
                BfFullTile scaled = TASSEMBLY<BfFullTile>(std::move(scaledParts));

                // --- encode to OutT and store data + both scales ---
                QuantTile<OutT> outFull;
                TCVT(outFull, scaled);              // bf16 -> OutT [32,64]

                global_iterator<gmData, QuantTile<OutT>> itD(
                    data_ptr + row0 * gN + kb * kBlock);
                auto gD = itD(0, 0);
                TSTORE(gD, outFull);

                global_iterator<gmScale, mq::ScaleCodeTile> itSA(
                    scale_ptr + row0 * numKb + kb);
                auto gSA = itSA(0, 0);
                TSTORE(gSA, scaleA);

                global_iterator<gmScale, mq::ScaleCodeTile> itSB(
                    scale_ptr + row0 * numKb + kb + 1);
                auto gSB = itSB(0, 0);
                TSTORE(gSB, scaleB);
            }
        } else {
            // Strided: each 32-wide block stored separately. gfrun/gfsim
            // runnable (no B.ASSEMBLE).
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < numKb; ++kb) {
                global_iterator<gmC, CBlockTile> itC(
                    c_scratch + row0 * gN + kb * kBlock);
                auto gC = itC(0, 0);
                CBlockTile cTile;
                TLOAD(cTile, gC);

                BfBlockTile bfTile;
                TCVT(bfTile, cTile);              // fp32 -> bf16

                BfBlockTile scaled;
                mq::ScaleCodeTile scaleCode;
                mq::quantize_block(scaled, scaleCode, bfTile);

                QuantHalfTile<OutT> outHalf;
                TCVT(outHalf, scaled);            // bf16 -> OutT [32,32]

                global_iterator<gmData, QuantHalfTile<OutT>> itD(
                    data_ptr + row0 * gN + kb * kBlock);
                auto gD = itD(0, 0);
                TSTORE(gD, outHalf);

                global_iterator<gmScale, mq::ScaleCodeTile> itS(
                    scale_ptr + row0 * numKb + kb);
                auto gS = itS(0, 0);
                TSTORE(gS, scaleCode);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stretch (not built here): a true HiF4 output = __fp4_hif4x2 data + U32
// scale word (E6M2 global scale + E1_8/E1_16 microexponents, group 64).
// Unimplementable today: E6M2 has no TileOP encode intrinsic (tohif4 uses a
// byte-cast stub, not a real quantize), and a U32 scale word is rejected by
// the emitter as an MX input. The structure above (group-32 amax -> scale ->
// reciprocal -> scale-mul -> pack) is the same algorithm; only the data/
// scale dtypes and the E6M2 encode step would change once the intrinsic
// exists. Mirror kernels/multi_thread/fa/tohif4 for the byte-cast stub form.
