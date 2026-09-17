#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// MXQuant [512,256] BF16 -> E4M3[512,256] + E8M0[512,8], CUBE_M32 layout.
//
// One operator, one shape: the four PEs split M (512 -> 128 rows each), and
// within a PE a 32-row block is quantized one 32-wide MX group at a time (8
// groups per block, 4 blocks per PE).
//
// |x| is folded as max(x, -x) rather than TABS.  On the current gfrun a unary
// TEPL op writes its CUBE_M32 destination densely instead of through the CELL
// indexer, which permutes the [32,32] tile (see LinxISA/SuperScalarModel issue
// #678).  TSUB/TMAX are CUBE-cell aware and max(x, -x) is bit-exact for every
// input, so the scale/quant result is identical to an abs-based one.
//
// The scale output is packed along N: the eight per-group E8M0 columns of a
// row block are TPACKed four-at-a-time into two U32 words per row, so each
// row block costs TWO 128 B scale stores (full CELLs) instead of eight
// 1/4-full ones.  The byte order is still plain row-major [512,8]
// (scale(row, group) = scales[row*8 + group]).
//
// No subview is used: each 32-wide group is TLOADed/TSTOREd with its own
// strided view into the [32,256] row block (the same trick the earlier
// half-block kernel used, generalised to eight columns).  Because the payload
// move is 64 B bursts at a 512 B stride (< the 256 B cacheline) this kernel is
// TLSU bound; the contiguous-load + TPARTVIEW + TASSEMBLY form that would fix
// it is blocked on the missing subview/assemble support (see the disabled
// variant at the bottom).

namespace mxquant {

using namespace pto;

constexpr int kTotalRows = 512;                       // complete tensor rows
constexpr int kRows = 32;                             // M32 tile rows
constexpr int kCols = 256;                            // complete tensor columns
constexpr int kBlock = 32;                            // MX group width
constexpr int kPeCount = 4;
constexpr int kRowsPerPe = kTotalRows / kPeCount;     // 128
constexpr int kRowBlocks = kRowsPerPe / kRows;        // 4 blocks of 32 rows
constexpr int kBlocksPerRow = kCols / kBlock;         // 8 MX groups per row
constexpr int kPackedWords = kBlocksPerRow / 4;       // 2 U32 words per row
constexpr int kE4M3Max = 448;                         // E4M3 max finite value

static_assert(kTotalRows % (kPeCount * kRows) == 0,
              "M must split into whole 32-row blocks per PE");
static_assert(kBlocksPerRow == 8, "this shape packs eight groups per row");
static_assert(kE4M3Max == 448, "this profile uses the E4M3 max finite value");

using BlockTile = VecTileM32<__bf16, kRows, kBlock>;
using RowMaxTile = VecTileM32<__bf16, kRows, 1>;
using QuantBlockTile = VecTileM32<__fp8_e4m3, kRows, kBlock>;
using ScaleTile = VecTileM32<__fp8_e8m0, kRows, 4, kRows, 1>;
using ScaleCodeTile = VecTileM32<uint8_t, kRows, 4, kRows, 1>;

static_assert(ScaleCodeTile::TilesizeCode == ScaleTile::TilesizeCode,
              "the u8 scale carrier must occupy the same Tile capacity as the "
              "E8M0 view of it");
static_assert(RowMaxTile::ValidCol == 1 && RowMaxTile::Cols == 1,
              "TROWMAX reduction result must be logical N x 1");

// One E8M0 scale column per MX block, carried in a [kRows, 4] byte tile whose
// physical columns match the 4-byte M32 cell width for 8-bit data.  Only the
// single valid column is used; the u8 carrier is required because the model
// tags each Tile register with the dtype of the block that last wrote it and a
// non-CUBE TSTORE's block dtype must equal that tag (a reinterpret_tile view
// cannot re-tag a register).

// BF16 amax -> E8M0 per OCP MX v1.0 Section 6.3: X = floor_pow2(amax) / 256.
// Since 256 = 2^8 this is scale_exp = floor(log2(amax)) - 8.
//
// The amax is converted DIRECTLY from BF16: the model's E8M0 encoder accepts
// BF16/FP16/FP32 sources (HardwareTCVTE8M0SourceTypeSupported), so the old
// FP16-widening TCVT is unnecessary and one [kRows,1] TCVT per MX group is
// dropped.  RTM (floor) still requires hand-written asm because the public
// TCVT is fixed at RNONE, whose E8M0 RNE midpoint is sqrt(2) and rounds up on
// (sqrt(2), 2) instead of flooring.  The destination is declared u8 and the
// E8M0 dtype is supplied to B.DATR directly, so the register is tagged u8 for
// the byte-domain arithmetic that follows.
inline void tcvt_e8m0_ocp(ScaleCodeTile &dst, RowMaxTile &src) {
  ScaleCodeTile amax_e8m0;
  asm volatile(
      "BSTART.TEPL 27, %D[SrcT]\n"
      "B.DATR %D[DstT], RTM\n"
      "B.DIM zero, %c[VCols], ->lb0\n"
      "B.DIM zero, %c[VRows], ->lb1\n"
      "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[Size]>\n"
      : [Dst] "=Tr"(amax_e8m0.data())
      : [SrcT] "i"(type_traits<__bf16>::TypeCode),
        [DstT] "i"(type_traits<__fp8_e8m0>::TypeCode),
        [Src] "Tr"(src.data()),
        [Size] "i"(ScaleCodeTile::TilesizeCode),
        [VCols] "i"(RowMaxTile::ValidCol),
        [VRows] "i"(RowMaxTile::ValidRow),
        [Cols] "i"(ScaleCodeTile::Cols)
      : "memory");

  // Step 2: apply the OCP /256 as an exponent subtraction in the raw byte
  // domain (256 = 2^8), rather than dividing the amax before the convert.  U8
  // TSUBS is ISA-legal (dtype-layout.asl:101) but gfrun's TEPL assertion was
  // narrower than the ASL until SuperScalarModel issue #625.
  TSUBS(dst, amax_e8m0, static_cast<uint8_t>(8));
}

// E8M0 is a biased-exponent byte, so its reciprocal is 254 - code in the raw
// byte domain: decode(254 - code) = 2^(127 - code) = 1 / decode(code).
inline void e8m0_reciprocal_code(ScaleCodeTile &dst, ScaleCodeTile &src) {
  ScaleCodeTile constant;
  TEXPANDS(constant, static_cast<uint8_t>(254));
  TSUB(dst, constant, src);
}

// Raw E8M0 codes (u8 carrier) -> BF16 values.  Hand-written asm declares the
// source dtype as E8M0 directly while binding the u8 tile's register.
inline void tcvt_e8m0_code_to_bf16(RowMaxTile &dst, ScaleCodeTile &src) {
  asm volatile(
      "BSTART.TEPL 27, %D[SrcT]\n"
      "B.DATR %D[DstT], RNONE\n"
      "B.DIM zero, %c[VCols], ->lb0\n"
      "B.DIM zero, %c[VRows], ->lb1\n"
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
  // |x| = max(x, -x): avoid TABS (see file header, issue #678).
  BlockTile zero_block;
  TSUB(zero_block, block, block);
  BlockTile neg_block;
  TSUB(neg_block, zero_block, block);
  BlockTile abs_block;
  TMAX(abs_block, block, neg_block);

  RowMaxTile amax;
  TROWMAX(amax, abs_block);

  tcvt_e8m0_ocp(scale, amax);

  ScaleCodeTile reciprocal_code;
  e8m0_reciprocal_code(reciprocal_code, scale);

  RowMaxTile reciprocal;
  tcvt_e8m0_code_to_bf16(reciprocal, reciprocal_code);

  TROWEXPANDMUL(dst, block, reciprocal);
}

// One 32-row x 32-column MX group: load, quantize, store the E4M3 payload, and
// hand back the E8M0 code column for the scale pack.
inline void group_pass(ScaleCodeTile &code, const __bf16 *in, __fp8_e4m3 *out) {
  BlockTile block;
  {
    global_tensor<__bf16, RowMajor<kRows, kCols>> gin(in);
    TLOAD(block, gin);
  }
  BlockTile scaled;
  quantize_block(scaled, code, block);

  QuantBlockTile quant;
  TCVT(quant, scaled);
  {
    global_tensor<__fp8_e4m3, RowMajor<kRows, kCols>> gout(out);
    TSTORE(gout, quant);
  }
}

// SPMD entry.  PE t owns rows [t*128, (t+1)*128).
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  const uint32_t tid = get_thread_idx();
  if (tid >= kPeCount) return;

  const __bf16 *my_input = input + tid * kRowsPerPe * kCols;
  __fp8_e4m3 *my_output = output + tid * kRowsPerPe * kCols;
  // Scale bytes are packed four groups per U32 word, so view the compact
  // [512,8] byte tensor as [512,2] U32.
  uint32_t *my_scale_words = reinterpret_cast<uint32_t *>(scales) +
                             tid * kRowsPerPe * kPackedWords;

  using WordTile = VecTileM32<uint32_t, kRows, 1>;

  for (int rb = 0; rb < kRowBlocks; ++rb) {
    const __bf16 *in_rb = my_input + rb * kRows * kCols;
    __fp8_e4m3 *out_rb = my_output + rb * kRows * kCols;

    ScaleCodeTile code0, code1, code2, code3, code4, code5, code6, code7;
    group_pass(code0, in_rb + 0 * kBlock, out_rb + 0 * kBlock);
    group_pass(code1, in_rb + 1 * kBlock, out_rb + 1 * kBlock);
    group_pass(code2, in_rb + 2 * kBlock, out_rb + 2 * kBlock);
    group_pass(code3, in_rb + 3 * kBlock, out_rb + 3 * kBlock);
    group_pass(code4, in_rb + 4 * kBlock, out_rb + 4 * kBlock);
    group_pass(code5, in_rb + 5 * kBlock, out_rb + 5 * kBlock);
    group_pass(code6, in_rb + 6 * kBlock, out_rb + 6 * kBlock);
    group_pass(code7, in_rb + 7 * kBlock, out_rb + 7 * kBlock);

    // Pack the eight E8M0 code columns, four at a time, into one U32 per row.
    // TCVT widens u8 -> U32 (value preserving); TPACK takes 1 byte from each
    // source, then 2 bytes from each, so word byte c is group c.
    WordTile w0, w1, w2, w3, w4, w5, w6, w7;
    TCVT(w0, code0);
    TCVT(w1, code1);
    TCVT(w2, code2);
    TCVT(w3, code3);
    TCVT(w4, code4);
    TCVT(w5, code5);
    TCVT(w6, code6);
    TCVT(w7, code7);

    WordTile p01, p23, low;
    TPACK(p01, w0, w1, 0x00000101);
    TPACK(p23, w2, w3, 0x00000101);
    TPACK(low, p01, p23, 0x00000202);  // groups 0..3

    WordTile p45, p67, high;
    TPACK(p45, w4, w5, 0x00000101);
    TPACK(p67, w6, w7, 0x00000101);
    TPACK(high, p45, p67, 0x00000202);  // groups 4..7

    // [32,2] U32 view with row stride 2 words = 8 bytes; low/high at columns
    // 0/1.  Together they fill the row block's 32 x 8 scale bytes.
    using ScaleWords = global_tensor<uint32_t, RowMajor<kRows, kPackedWords>>;
    uint32_t *row_block_words = my_scale_words + rb * kRows * kPackedWords;

    global_iterator<ScaleWords, WordTile> low_iter(row_block_words);
    auto low_global = low_iter(0, 0);
    TSTORE(low_global, low);

    global_iterator<ScaleWords, WordTile> high_iter(row_block_words + 1);
    auto high_global = high_iter(0, 0);
    TSTORE(high_global, high);
  }
}

#if 0
// ---------------------------------------------------------------------------
// Subview variant -- BLOCKED on LinxISA/Linx-TileOP-API issue #144.
//
// Goal: replace the eight strided [32,32] payload TLOADs per row block (64 B
// bursts at a 512 B stride, ~12.5% cacheline utilisation) with ONE contiguous
// [32,256] TLOAD (rows 512 B >= the 256 B cacheline) plus eight [32,32]
// subviews for the per-group reductions.
//
// It does not compile today: TPARTVIEW's element is region::SubTileView, which
// is not recognised by is_tile / is_tile_data_v, so TROWMIN and TROWEXPANDMUL
// reject it (only TROWMAX / TEXP accept it).  range::Subview is a legal operand
// but forwards the parent [32,256] shape instead of the fragment [32,32].  See
// the issue for the full support matrix.  Delete the #if 0 once region::
// SubTileView is a general tile operand (with its own valid shape).
//
// Sketch (per PE, per 32-row block rb):
//   using RowBlockTile = VecTileM32<__bf16, kRows, kCols>;      // [32,256]
//   RowBlockTile row_block;
//   TLOAD(row_block, global_tensor<__bf16, RowMajor<kRows, kCols>>(in_rb));
//   auto parts = TPARTVIEW<BlockTile, 1, kBlocksPerRow>(row_block);
//   for g in [0,8) (unrolled, keeping code0..code7 for the scale pack):
//     TROWMAX(row_max, parts[0][g]);
//     TROWMIN(row_min, parts[0][g]);
//     TSUB(zero, row_min, row_min);
//     TSUB(neg_min, zero, row_min);
//     TMAX(amax, row_max, neg_min);
//     ... E8M0 code + reciprocal (same as quantize_block) ...
//     TROWEXPANDMUL(scaled, parts[0][g], reciprocal);   // subview as src0
//     TCVT(quant, scaled);
//     TSTORE(per-group strided view);   // still 32 B / 256 B: the store cannot
//                                       // be made one [32,256] row until
//                                       // TASSEMBLY/B.ASSEMBLE is modelled.
//
// Store side: assembling the eight quantised groups back into one [32,256] row
// and TSTOREing it whole needs TASSEMBLY / B.ASSEMBLE; gfsim does not model
// B.ASSEMBLE yet.
#endif

}  // namespace mxquant
