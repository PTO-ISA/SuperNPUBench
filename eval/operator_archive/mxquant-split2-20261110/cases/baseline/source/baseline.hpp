#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

// MXQuant [512,256] BF16 -> E4M3[512,256] + E8M0[512,8], CUBE_M32 layout.
//
// One operator, one shape: the four PEs split M (512 -> 128 rows each), and
// within a PE a 32-row block is quantized one 32-wide MX group at a time (8
// groups per block, 4 blocks per PE).  Everything stays CUBE_M32.
//
// |x| uses TABS directly.  The earlier #678 concern (a unary TEPL writing a
// CUBE_M32 destination densely, permuting the tile) does not reproduce on the
// current main gfrun: the TABS variant is byte-exact here.
//
// PTO #311 row reduction: a CUBE_M32 rowmax keeps its [32,1] result in the
// prefix CELL of a wide carrier (RowMaxWideTile) whose physical columns equal
// the source's; TREDUCEPREFIXVIEW exposes that CELL zero-copy and the public
// TCVT reduction-prefix overload consumes it.
//
// E8M0 scale generation is public TileOP only:
//   * BF16 -> E8M0 uses `TCVT<LINX_RDN>` (RTM floor, LinxRMode 3).  OCP /256 is
//     folded as a 2^-8 BF16 pre-scale, so no byte-domain E8M0 arithmetic is
//     needed (gfrun rejects E8M0 scalar arithmetic, and a U8 carrier
//     reinterpreted as E8M0 would violate the CUBE-source backing-dtype
//     contract -- PTO-ISA pto-spec issue #324).
//   * E8M0 -> BF16 uses `TCVT` from a genuinely E8M0-backed source, then
//     TRECIP in BF16 (exact for power-of-two scales).
//
// IMPORTANT: the public TLOAD/TSTORE (and other TLSU) wrappers in
// Linx-TileOP-API carry no `: "memory"` clobber, so the backend may reorder the
// GM accesses / address setup and silently produce "values right, landing
// wrong".  The barriers below are the workaround (verified on the pure-copy
// `test/kernel/mxprobe` probe); they can be dropped once the toolchain adds the
// clobber.

namespace mxquant {

using namespace pto;

constexpr int kTotalRows = 512;
constexpr int kRows = 32;
constexpr int kCols = 256;
constexpr int kBlock = 32;
constexpr int kPeCount = 4;
constexpr int kRowsPerPe = kTotalRows / kPeCount;   // 128
constexpr int kRowBlocks = kRowsPerPe / kRows;      // 4
constexpr int kBlocksPerRow = kCols / kBlock;       // 8
constexpr int kE4M3Max = 448;

static_assert(kTotalRows % (kPeCount * kRows) == 0,
              "M must split into whole 32-row blocks per PE");
static_assert(kBlocksPerRow == 8, "this shape has eight groups per row");
static_assert(kE4M3Max == 448, "this profile uses the E4M3 max finite value");

using BlockTile = VecTileM32<__bf16, kRows, kBlock>;
// PTO #311 carrier: same physical columns as the source, ValidCol == 1.
using RowMaxWideTile = VecTileM32<__bf16, kRows, kBlock, kRows, 1>;
// Compact descriptor consumed by the TCVT reduction-prefix overload.
using RowMaxTile = VecTileM32<__bf16, kRows, 1, kRows, 1>;
using QuantBlockTile = VecTileM32<__fp8_e4m3, kRows, kBlock>;
using ScaleTile = VecTileM32<__fp8_e8m0, kRows, 4, kRows, 1>;

static_assert(RowMaxTile::ValidCol == 1 && RowMaxTile::Cols == 1,
              "TROWMAX reduction result must be a one-column valid row");
static_assert(ScaleTile::ValidCol == 1,
              "E8M0 scale carrier holds one valid column");

// 2^-8 in BF16: biased exponent 127 - 8 = 119 -> 0x3B80.
inline constexpr __bf16 kInv256 =
    __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x3B80));

#define MXQ_BARRIER() asm volatile("" ::: "memory")

// Quantize one 32-wide MX block, emit its per-row E8M0 scale code, and return
// the scaled block.  OCP MX v1.0 section 6.3: X = floor_pow2(amax) / 256.
inline void quantize_block(BlockTile &dst, ScaleTile &scale, BlockTile &block) {
  BlockTile abs_block;
  TABS(abs_block, block);
  MXQ_BARRIER();

  // OCP /256 folded as a 2^-8 pre-scale (exact for normal BF16).
  BlockTile abs_scaled;
  TMULS(abs_scaled, abs_block, kInv256);
  MXQ_BARRIER();

  // PTO #311 wide-carrier rowmax + zero-copy prefix view.
  RowMaxWideTile amax_wide;
  TROWMAX(amax_wide, abs_scaled);
  MXQ_BARRIER();
  auto amax_view = TREDUCEPREFIXVIEW<RowMaxTile>(amax_wide);

  // RTM floor is mandatory (RNE midpoint sqrt(2) would round up).
  TCVT<LINX_RDN>(scale, amax_view);
  MXQ_BARRIER();

  // Decode E8M0 -> BF16 and invert for the payload multiply.
  RowMaxTile scale_bf16;
  TCVT(scale_bf16, scale);
  RowMaxTile reciprocal;
  TRECIP(reciprocal, scale_bf16);
  MXQ_BARRIER();

  TROWEXPANDMUL(dst, block, reciprocal);
  MXQ_BARRIER();
}

// One 32-row x 32-column MX group: load, quantize, store the E4M3 payload, and
// hand back the E8M0 code column for the scale store.
inline void group_pass(ScaleTile &code, const __bf16 *in, __fp8_e4m3 *out) {
  BlockTile block;
  {
    global_tensor<__bf16, RowMajor<kRows, kCols>> gin(in);
    TLOAD(block, gin);
  }
  MXQ_BARRIER();
  BlockTile scaled;
  quantize_block(scaled, code, block);

  QuantBlockTile quant;
  TCVT(quant, scaled);
  MXQ_BARRIER();
  {
    global_tensor<__fp8_e4m3, RowMajor<kRows, kCols>> gout(out);
    TSTORE(gout, quant);
  }
  MXQ_BARRIER();
}

// SPMD entry.  PE t owns rows [t*128, (t+1)*128).  Scale bytes are row-major
// [512,8] (scale(row, group) = scales[row*8 + group]).
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  const uint32_t tid = get_thread_idx();
  if (tid >= kPeCount) return;

  const __bf16 *my_input = input + tid * kRowsPerPe * kCols;
  __fp8_e4m3 *my_output = output + tid * kRowsPerPe * kCols;
  __fp8_e8m0 *my_scales = scales + tid * kRowsPerPe * kBlocksPerRow;

  using ScaleRowBlock =
      global_tensor<__fp8_e8m0, RowMajor<kRows, kBlocksPerRow>>;

  for (int rb = 0; rb < kRowBlocks; ++rb) {
    const __bf16 *in_rb = my_input + rb * kRows * kCols;
    __fp8_e4m3 *out_rb = my_output + rb * kRows * kCols;
    __fp8_e8m0 *scale_rb = my_scales + rb * kRows * kBlocksPerRow;

    ScaleTile code0, code1, code2, code3, code4, code5, code6, code7;
    group_pass(code0, in_rb + 0 * kBlock, out_rb + 0 * kBlock);
    group_pass(code1, in_rb + 1 * kBlock, out_rb + 1 * kBlock);
    group_pass(code2, in_rb + 2 * kBlock, out_rb + 2 * kBlock);
    group_pass(code3, in_rb + 3 * kBlock, out_rb + 3 * kBlock);
    group_pass(code4, in_rb + 4 * kBlock, out_rb + 4 * kBlock);
    group_pass(code5, in_rb + 5 * kBlock, out_rb + 5 * kBlock);
    group_pass(code6, in_rb + 6 * kBlock, out_rb + 6 * kBlock);
    group_pass(code7, in_rb + 7 * kBlock, out_rb + 7 * kBlock);

    // One strided [32,1] E8M0 store per group into the [512,8] byte tensor.
    {
      ScaleRowBlock gs(scale_rb + 0);
      TSTORE(gs, code0);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 1);
      TSTORE(gs, code1);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 2);
      TSTORE(gs, code2);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 3);
      TSTORE(gs, code3);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 4);
      TSTORE(gs, code4);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 5);
      TSTORE(gs, code5);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 6);
      TSTORE(gs, code6);
    }
    MXQ_BARRIER();
    {
      ScaleRowBlock gs(scale_rb + 7);
      TSTORE(gs, code7);
    }
    MXQ_BARRIER();
  }
}

#undef MXQ_BARRIER

}  // namespace mxquant
