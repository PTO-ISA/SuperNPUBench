#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// MXQuant: BF16[32,64] -> E4M3[32,64] + E8M0[32,2].
//
// Each row is split into two 32-wide MX blocks.  Per block g:
//
//   amax[g]       = rowmax(abs(X[:, g*32 : g*32+32]))      // BF16[32,1]
//   scale[g]      = MX_E8M0(amax[g])                       // covers QMAX=448
//   Q[:, block g] = E4M3_RNE_SAT(X[:, block g] * (1/scale[g]))
//
// The two blocks arrive as separate [32,32] tiles (the test loads them with
// two strided TLOADs).  The scaled halves are encoded to E4M3 independently
// and stored back with two strided TSTOREs; a TASSEMBLY variant that packs
// them into one [32,64] tile first is kept disabled below (gfsim does not
// model B.ASSEMBLE; TPARTVIEW subview sources are only modelled for CUBE
// layouts, so RowMajor operands use materialized tiles instead).
//
// Two ISA representation rules shape the helper types below:
//   - TROWMAX/TROWEXPANDMUL require their reduction result / broadcast
//     operand to be exactly N x 1, hence RowMaxTile.
//   - Ordinary TCVT requires equal capacity-derived physical rows and equal
//     ValidRow/ValidCol on both sides.  A narrow BF16<->E8M0 pair therefore
//     carries the single scale column in a [32,2] byte tile with ValidCol=1
//     (ScaleTile); see the TCVT legality note in template_asm.hpp.

namespace mxquant {

using namespace pto;

constexpr int kTotalRows = 128;          // Complete tensor: 4 PE × 32
constexpr int kRows = 32;                // Rows per PE
constexpr int kCols = 64;
constexpr int kBlock = 32;
constexpr int kBlocksPerRow = kCols / kBlock;
constexpr int kE4M3Max = 448;

// Layout note: Vec + CubeM32 (VecTileM32) was tried and reverted — the API
// rejects CubeM16/M32 fractals as TCVT operands unless the tile has a Matrix
// location (template_asm.hpp "TCVT CUBE_M16/M32 destination must have Matrix
// location"), so Vec M32 tiles can only be TLOADed/TSTOREd, not converted.
// Until the API supports Vec-side M32 TCVT, compute tiles stay RowMajor.
using InputTile = Tile<Location::Vec, __bf16, kRows, kCols, BLayout::RowMajor>;
using BlockTile = Tile<Location::Vec, __bf16, kRows, kBlock, BLayout::RowMajor>;
using RowMaxTile = Tile<Location::Vec, __bf16, kRows, 1, BLayout::RowMajor>;
// The RTM amax -> E8M0 convert below is written as FP16 -> E8M0, so the BF16
// row max is widened first.  BF16->FP16 is an exact widening for this
// kernel's amplitude range.  (Narrowing this to a direct BF16 -> E8M0 RTM
// convert would drop one TCVT, but that source/dest pair is unverified
// against the model.)
using Fp16RowMaxTile = Tile<Location::Vec, __half, kRows, 1, BLayout::RowMajor>;
using QuantizedTile =
    Tile<Location::Vec, __fp8_e4m3, kRows, kCols, BLayout::RowMajor>;
// One quantized MX block; run() stores the halves with strided TSTOREs
// (gfsim does not model B.ASSEMBLE).
using QuantBlockTile =
    Tile<Location::Vec, __fp8_e4m3, kRows, kBlock, BLayout::RowMajor>;

// One E8M0 scale column per MX block.  Stored as [32,2] with ValidCol=1 so
// BF16<->E8M0 TCVT keeps matching capacity-derived rows; TSTORE writes only
// the valid column, so the GM layout stays the compact [32,2] form.
using ScaleTile =
    Tile<Location::Vec, __fp8_e8m0, kRows, 2, BLayout::RowMajor, kRows, 1>;

// Same geometry and bytes as ScaleTile, but a native uint8_t tile: this is the
// carrier the E8M0 exponent codes actually live in while they are being
// manipulated as raw bytes (TSUBS/TSUB/TEXPANDS) and when they are stored.
//
// It has to be a real u8 tile rather than a reinterpret_tile view of a
// ScaleTile, for two reasons:
//   - TSUBS(dst, src, s) binds dst and src to one template parameter, so a
//     native tile and a view of it are different types and will not bind.
//   - The model tags each Tile register with the dtype of the block that last
//     wrote it.  A byte-domain write leaves the register tagged UINT8, and
//     ValidateLocalTlsu (AccumulateBlockInfo.cpp:190) requires a non-CUBE
//     TSTORE's block dtype to equal that tag -- so the store must be a u8
//     store.  reinterpret_tile cannot fix this at the store: it is a
//     zero-instruction compile-time view, so it never re-tags the register,
//     and its view type does not expose IsCubeLayout, which TSTORE requires.
// E8M0 and uint8_t are both 8 bit and TSTORE does not convert, so the bytes
// landing in GM are identical either way.
using ScaleCodeTile =
    Tile<Location::Vec, uint8_t, kRows, 2, BLayout::RowMajor, kRows, 1>;

static_assert(ScaleCodeTile::TilesizeCode == ScaleTile::TilesizeCode,
              "the u8 scale carrier must occupy the same Tile capacity as the "
              "E8M0 view of it");

static_assert(kCols % kBlock == 0, "MX block must divide the input width");
static_assert(kE4M3Max == 448, "this profile uses the E4M3 max finite value");
static_assert(RowMaxTile::ValidCol == 1 && RowMaxTile::Cols == 1,
              "TROWMAX reduction result must be logical N x 1");

// BF16 amax -> E8M0 per OCP MX v1.0 Section 6.3: X = floor_pow2(amax) / 256.
// Since 256 = 2^8, this is equivalent to: scale_exp = floor(log2(amax)) - 8.
// Writes the raw E8M0 exponent code into a u8 carrier; see ScaleCodeTile for
// why the code stays in a u8 tile rather than an E8M0 one.
inline void tcvt_e8m0_ocp(ScaleCodeTile &dst, Fp16RowMaxTile &src) {
  // Step 1: unscaled FP16 amax -> E8M0 with RTM (floor) rounding.  Public
  // TCVT is fixed at RNONE, whose E8M0 RNE midpoint is the geometric mean
  // sqrt(2), so it rounds up throughout (sqrt(2), 2) instead of flooring;
  // hand-written asm is the only way to request RTM.  The destination is
  // declared u8 and the E8M0 dtype is supplied to B.DATR directly, so the
  // register is tagged u8 for the byte-domain arithmetic that follows.
  ScaleCodeTile amax_e8m0;
  asm volatile(
      "BSTART.TEPL 27, %D[SrcT]\n"
      "B.DATR %D[DstT], RTM\n"  // RTM = floor (LinxRMode 3)
      "B.DIM zero, %c[VCols], ->lb0\n"
      "B.DIM zero, %c[VRows], ->lb1\n"
      "B.DIM zero, %c[Cols], ->lb2\n"
      "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[Size]>\n"
      : [Dst] "=Tr"(amax_e8m0.data())
      : [SrcT] "i"(type_traits<__half>::TypeCode),
        [DstT] "i"(type_traits<__fp8_e8m0>::TypeCode),
        [Src] "Tr"(src.data()),
        [Size] "i"(ScaleCodeTile::TilesizeCode),
        [VCols] "i"(Fp16RowMaxTile::ValidCol),
        [VRows] "i"(Fp16RowMaxTile::ValidRow),
        [Cols] "i"(ScaleCodeTile::Cols)
      : "memory");

  // Step 2: apply the OCP /256 as an exponent subtraction in the raw byte
  // domain (256 = 2^8).  Preferred over dividing the FP16 amax by 256 before
  // the convert: no FP16 rounding step at all, so it stays exact for
  // amplitudes an FP16 division would have flushed toward subnormal.  U8
  // TSUBS is ISA-legal -- TileVecArithmeticDataTypeSupported
  // (dtype-layout.asl:101) admits U8 -- but gfrun's TEPL assertion was
  // narrower than the ASL until SuperScalarModel issue #625.
  // Caveat: this is a byte-domain subtract, so an amax whose E8M0 exponent
  // code is below 8 wraps instead of clamping.  Out of range for this
  // kernel's inputs; a saturating form would need TMAX against 8 first.
  TSUBS(dst, amax_e8m0, static_cast<uint8_t>(8));
}

// E8M0 is a biased-exponent byte, so its reciprocal is 254 - code in the raw
// byte domain: decode(254 - code) = 2^(127 - code) = 1 / decode(code).
// TSUB on E8M0 tiles would be a numeric subtraction, not a storage-byte one,
// so the codes are kept in u8 carriers throughout.
inline void e8m0_reciprocal_code(ScaleCodeTile &dst, ScaleCodeTile &src) {
  ScaleCodeTile constant;
  TEXPANDS(constant, static_cast<uint8_t>(254));
  TSUB(dst, constant, src);
}

// Decode raw E8M0 exponent codes held in a u8 carrier into BF16 values.
//
// The public TCVT cannot express this: its source would have to be a
// reinterpret_tile<__fp8_e8m0> view, and that view type does not expose
// StorageBytes, which TCVT's derived-rows check reads.  Hand-written asm
// instead declares the source dtype as E8M0 directly while binding the u8
// tile's register -- the same technique tcvt_e8m0_ocp uses on the way in.
//
// The emitted encoding is identical to what TCVT(RowMaxTile, ScaleTile) would
// produce: LB0/LB1 carry the source's valid columns/rows, LB2 the destination's
// physical columns, and TSize the destination's capacity.  The [32,2]
// ValidCol=1 source geometry is what keeps the capacity-derived physical rows
// equal on both sides.
inline void tcvt_e8m0_code_to_bf16(RowMaxTile &dst, ScaleCodeTile &src) {
  asm volatile(
      "BSTART.TEPL 27, %D[SrcT]\n"
      "B.DATR %D[DstT], RNONE\n"
      "B.DIM zero, %c[VCols], ->lb0\n"
      "B.DIM zero, %c[VRows], ->lb1\n"
      "B.DIM zero, %c[Cols], ->lb2\n"
      "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[Size]>\n"
      : [Dst] "=Tr"(dst.data())
      : [SrcT] "i"(type_traits<__fp8_e8m0>::TypeCode),
        [DstT] "i"(type_traits<__bf16>::TypeCode),
        [Src] "Tr"(src.data()),
        [Size] "i"(RowMaxTile::TilesizeCode),
        [VCols] "i"(ScaleCodeTile::ValidCol),
        [VRows] "i"(ScaleCodeTile::ValidRow),
        [Cols] "i"(RowMaxTile::Cols)
      : "memory");
}

// Quantize one 32-wide MX block and emit its per-row E8M0 scale code.
inline void quantize_block(BlockTile &dst, ScaleCodeTile &scale,
                           BlockTile &block) {
  BlockTile abs_block;
  TABS(abs_block, block);

  RowMaxTile amax;
  TROWMAX(amax, abs_block);

  // OCP MX v1.0 §6.3: scale = floor_pow2(amax) / 256.
  // The RTM convert in tcvt_e8m0_ocp takes an FP16 source, so widen first
  // (exact for this amplitude range); the /256 is applied there as a
  // byte-domain exponent subtract.
  Fp16RowMaxTile amax_h;
  TCVT(amax_h, amax);

  tcvt_e8m0_ocp(scale, amax_h);

  ScaleCodeTile reciprocal_code;
  e8m0_reciprocal_code(reciprocal_code, scale);

  RowMaxTile reciprocal;
  tcvt_e8m0_code_to_bf16(reciprocal, reciprocal_code);

  TROWEXPANDMUL(dst, block, reciprocal);
}

// SPMD entry: [128, 64] complete tensor, 4 PE each process 32 rows.
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  const uint32_t tid = get_thread_idx();
  if (tid >= 4) return;

  // Each PE gets a contiguous [32, 64] row segment
  const __bf16 *my_input = input + tid * kRows * kCols;
  __fp8_e4m3 *my_output = output + tid * kRows * kCols;
  __fp8_e8m0 *my_scales = scales + tid * kRows * kBlocksPerRow;
  // The scale codes are stored from a u8 tile, so the GM view is u8 too; the
  // bytes are identical to the E8M0 ones the caller expects.
  uint8_t *my_scale_codes = reinterpret_cast<uint8_t *>(my_scales);

  // Process two [32, 32] half-blocks with strided TLOAD/TSTORE
  global_tensor<__bf16, RowMajor<kRows, kCols>> input_tensor(my_input);
  global_tensor<__fp8_e4m3, RowMajor<kRows, kCols>> output_tensor(my_output);

  // Left half-block [32, 32]: columns [0, 32)
  BlockTile left_block;
  TLOAD(left_block, input_tensor);

  BlockTile scaled_left;
  ScaleCodeTile left_scale;
  quantize_block(scaled_left, left_scale, left_block);

  QuantBlockTile left_out;
  TCVT(left_out, scaled_left);
  TSTORE(output_tensor, left_out);

  global_iterator<global_tensor<uint8_t, RowMajor<kRows, kBlocksPerRow>>, ScaleCodeTile>
      left_scale_iter(my_scale_codes);
  auto left_scale_global = left_scale_iter(0, 0);
  TSTORE(left_scale_global, left_scale);

  // Right half-block [32, 32]: columns [32, 64)
  global_tensor<__bf16, RowMajor<kRows, kCols>> input_right(my_input + kBlock);
  BlockTile right_block;
  TLOAD(right_block, input_right);

  BlockTile scaled_right;
  ScaleCodeTile right_scale;
  quantize_block(scaled_right, right_scale, right_block);

  QuantBlockTile right_out;
  TCVT(right_out, scaled_right);

  global_tensor<__fp8_e4m3, RowMajor<kRows, kCols>> output_right(my_output + kBlock);
  TSTORE(output_right, right_out);

  global_iterator<global_tensor<uint8_t, RowMajor<kRows, kBlocksPerRow>>, ScaleCodeTile>
      right_scale_iter(my_scale_codes + 1);
  auto right_scale_global = right_scale_iter(0, 0);
  TSTORE(right_scale_global, right_scale);
}

#if 0
// gfrun-only variant: pack the scaled halves back into one [kRows, kCols]
// tile before the final E4M3 encode.  Kept for reference — TASSEMBLY lowers
// to B.ASSEMBLE, which gfsim does not model, so the split-store run() above
// is the default.
inline void run(QuantizedTile &output, ScaleTile &left_scale,
                ScaleTile &right_scale, BlockTile &left, BlockTile &right) {
  BlockTile scaled_left, scaled_right;
  quantize_block(scaled_left, left_scale, left);
  quantize_block(scaled_right, right_scale, right);

  TileArray<BlockTile, 1, kBlocksPerRow> scaled_parts;
  TCVT(scaled_parts[0][0], scaled_left);
  TCVT(scaled_parts[0][1], scaled_right);
  InputTile scaled = TASSEMBLY<InputTile>(std::move(scaled_parts));

  // BF16 -> E4M3, round-to-nearest-even with saturation.
  TCVT(output, scaled);
}
#endif

}  // namespace mxquant
