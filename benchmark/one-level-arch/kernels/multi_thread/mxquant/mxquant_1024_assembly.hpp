#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// MXQuant Assembly: BF16[32,1024] -> E4M3[32,1024] + E8M0[32,32].
//
// Strategy: Load all 32 blocks, quantize in batch, then store.
// gfsim-compatible (TASSEMBLY path commented out, uses strided TSTORE).
//
// Each row is split into 32 MX blocks. Per block g:
//   amax[g]       = rowmax(abs(X[:, g*32 : g*32+32]))
//   scale[g]      = MX_E8M0(amax[g])
//   Q[:, block g] = E4M3_RNE_SAT(X[:, block g] * (1/scale[g]))

namespace mxquant_assembly {

using namespace pto;

constexpr int kTotalRows = 128;          // Complete tensor: 4 PE × 32
constexpr int kRows = 32;                // Rows per PE
constexpr int kCols = 1024;
constexpr int kBlock = 32;
constexpr int kBlocksPerRow = kCols / kBlock;  // = 32
constexpr int kE4M3Max = 448;

using BlockTile = Tile<Location::Vec, __bf16, kRows, kBlock, BLayout::RowMajor>;
using RowMaxTile = Tile<Location::Vec, __bf16, kRows, 1, BLayout::RowMajor>;
using Fp16RowMaxTile = Tile<Location::Vec, __half, kRows, 1, BLayout::RowMajor>;
using QuantBlockTile = Tile<Location::Vec, __fp8_e4m3, kRows, kBlock, BLayout::RowMajor>;
using ScaleTile = Tile<Location::Vec, __fp8_e8m0, kRows, 2, BLayout::RowMajor, kRows, 1>;

// Same geometry and bytes as ScaleTile, but a native uint8_t tile: this is the
// carrier the E8M0 exponent codes live in while they are manipulated as raw
// bytes (TSUBS/TSUB/TEXPANDS) and when they are stored.  It has to be a real
// u8 tile rather than a reinterpret_tile view, because the model tags each
// Tile register with the dtype of the block that last wrote it and
// ValidateLocalTlsu (AccumulateBlockInfo.cpp:190) requires a non-CUBE TSTORE's
// block dtype to equal that tag.  See mxquant.hpp for the full rationale.
using ScaleCodeTile = Tile<Location::Vec, uint8_t, kRows, 2, BLayout::RowMajor, kRows, 1>;

static_assert(ScaleCodeTile::TilesizeCode == ScaleTile::TilesizeCode,
              "the u8 scale carrier must occupy the same Tile capacity as the "
              "E8M0 view of it");

static_assert(kCols % kBlock == 0, "MX block must divide input width");
static_assert(kE4M3Max == 448, "E4M3 max finite value");

// BF16 amax -> E8M0 per OCP MX v1.0 Section 6.3: X = floor_pow2(amax) / 256.
// Since 256 = 2^8, this is equivalent to: scale_exp = floor(log2(amax)) - 8.
// Hand-written asm with RTM (round toward minus infinity = floor).
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
  // domain (256 = 2^8).  No FP16 rounding step at all, so it stays exact for
  // amplitudes an FP16 division would have flushed toward subnormal.  U8
  // TSUBS is ISA-legal -- TileVecArithmeticDataTypeSupported
  // (dtype-layout.asl:101) admits U8 -- but gfrun's TEPL assertion was
  // narrower than the ASL until SuperScalarModel issue #625.
  // Caveat: byte-domain subtract, so an amax whose E8M0 exponent code is below
  // 8 wraps instead of clamping.  Out of range for this kernel's inputs.
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
// Public TCVT cannot express this: its source would have to be a
// reinterpret_tile<__fp8_e8m0> view, and that view type does not expose
// StorageBytes, which TCVT's derived-rows check reads.  See mxquant.hpp.
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

// Quantize one 32-wide MX block
inline void quantize_block(BlockTile &dst, ScaleCodeTile &scale, BlockTile &block) {
  BlockTile abs_block;
  TABS(abs_block, block);

  RowMaxTile amax;
  TROWMAX(amax, abs_block);

  // OCP MX v1.0 §6.3: scale = floor_pow2(amax) / 256
  Fp16RowMaxTile amax_h;
  TCVT(amax_h, amax);

  tcvt_e8m0_ocp(scale, amax_h);

  ScaleCodeTile reciprocal_code;
  e8m0_reciprocal_code(reciprocal_code, scale);

  RowMaxTile reciprocal;
  tcvt_e8m0_code_to_bf16(reciprocal, reciprocal_code);

  TROWEXPANDMUL(dst, block, reciprocal);
}

// SPMD + Batch run: [128, 1024] complete tensor, 4 PE each process 32 rows.
// Each PE processes blocks in groups of 8 to stay under 64KB tile frame limit.
// Peak liveness per batch: 8×BlockTile (16KB) + 8×scaled (16KB) + 8×ScaleTile
// (~0.5KB) ≈ 32.5KB, well below the compiler's 64KB constraint.
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  const uint32_t tid = get_thread_idx();
  if (tid >= 4) return;

  // Each PE gets a contiguous [32, 1024] row segment
  const __bf16 *my_input = input + tid * kRows * kCols;
  __fp8_e4m3 *my_output = output + tid * kRows * kCols;
  __fp8_e8m0 *my_scales = scales + tid * kRows * kBlocksPerRow;
  // The scale codes are stored from a u8 tile, so the GM view is u8 too; the
  // bytes are identical to the E8M0 ones the caller expects.
  uint8_t *my_scale_codes = reinterpret_cast<uint8_t *>(my_scales);

  constexpr int kBatchSize = 8;
  constexpr int kNumBatches = kBlocksPerRow / kBatchSize;  // 32 / 8 = 4
  static_assert(kBlocksPerRow % kBatchSize == 0, "kBlocksPerRow must be divisible by kBatchSize");

  for (int batch = 0; batch < kNumBatches; ++batch) {
    const int batch_start = batch * kBatchSize;

    // Load one batch of blocks from this PE's row segment
    BlockTile blocks[kBatchSize];
    #pragma unroll
    for (int i = 0; i < kBatchSize; ++i) {
      const int block_idx = batch_start + i;
      global_tensor<__bf16, RowMajor<kRows, kCols>> source(my_input + block_idx * kBlock);
      TLOAD(blocks[i], source);
    }

    // Quantize this batch
    BlockTile scaled_blocks[kBatchSize];
    ScaleCodeTile scale_tiles[kBatchSize];
    #pragma unroll
    for (int i = 0; i < kBatchSize; ++i) {
      quantize_block(scaled_blocks[i], scale_tiles[i], blocks[i]);
    }

    // Convert and store this batch to this PE's row segment (gfsim-compatible, no TASSEMBLY)
    #pragma unroll
    for (int i = 0; i < kBatchSize; ++i) {
      const int block_idx = batch_start + i;

      QuantBlockTile quantized_block;
      TCVT(quantized_block, scaled_blocks[i]);

      global_tensor<__fp8_e4m3, RowMajor<kRows, kCols>>
          output_block(my_output + block_idx * kBlock);
      TSTORE(output_block, quantized_block);

      global_tensor<uint8_t, RowMajor<kRows, kBlocksPerRow>>
          scale_out(my_scale_codes + block_idx);
      TSTORE(scale_out, scale_tiles[i]);
    }
  }
}

}  // namespace mxquant_assembly
