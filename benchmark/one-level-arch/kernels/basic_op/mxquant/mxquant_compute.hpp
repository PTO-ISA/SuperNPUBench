#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "basic_op/mxquant/mxquant.hpp"

// Compute-only (no TLSU) siblings of mxquant, for performance work.
//
// The full kernel is TLSU bound, so these variants keep the exact Vector/TEPL
// stream but remove every TLOAD/TSTORE: the input is assumed to already live in
// the register file and is materialised with a single TEXPANDS, and nothing is
// stored back.  Every op lowers to `asm volatile`, so nothing is dead-code
// eliminated even though the results are unused.
//
// These are TIMING-ONLY builds: they write no output, so res_check cannot use
// them, and the kTabs=true instance is numerically invalid on gfrun anyway
// (unary TEPL writes the CUBE_M32 destination densely instead of through the
// CELL indexer -- LinxISA/SuperScalarModel issue #678).  gfsim does not
// evaluate data, so it still times them.
//
//   kTabs=false  |x| = max(x, -x)   model-safe, matches mxquant
//   kTabs=true   |x| = TABS         ideal instruction count (removes two ops
//                                   per MX group) -- timing only
//   kExpand      whether the input TEXPANDS is present; kExpand=false drops it
//                so the TEXPANDS cost can be isolated by subtraction

namespace mxquant_compute {

using namespace pto;

using mxquant::BlockTile;
using mxquant::RowMaxTile;
using mxquant::QuantBlockTile;
using mxquant::ScaleCodeTile;
using mxquant::tcvt_e8m0_ocp;
using mxquant::e8m0_reciprocal_code;
using mxquant::tcvt_e8m0_code_to_bf16;

constexpr int kTotalRows = mxquant::kTotalRows;       // 512
constexpr int kRows = mxquant::kRows;                 // 32
constexpr int kCols = mxquant::kCols;                 // 256
constexpr int kBlock = mxquant::kBlock;               // 32
constexpr int kPeCount = mxquant::kPeCount;           // 4
constexpr int kRowsPerPe = mxquant::kRowsPerPe;       // 128
constexpr int kRowBlocks = mxquant::kRowBlocks;       // 4
constexpr int kBlocksPerRow = mxquant::kBlocksPerRow; // 8

// quantize one 32-wide MX block, selecting the |x| fold.
template <bool kTabs>
inline void quantize(BlockTile &dst, ScaleCodeTile &scale, BlockTile &block) {
  if constexpr (!kTabs) {
    mxquant::quantize_block(dst, scale, block);
  } else {
    BlockTile abs_block;
    TABS(abs_block, block);

    RowMaxTile amax;
    TROWMAX(amax, abs_block);

    tcvt_e8m0_ocp(scale, amax);

    ScaleCodeTile reciprocal_code;
    e8m0_reciprocal_code(reciprocal_code, scale);

    RowMaxTile reciprocal;
    tcvt_e8m0_code_to_bf16(reciprocal, reciprocal_code);

    TROWEXPANDMUL(dst, block, reciprocal);
  }
}

// One MX group: materialise (optionally), quantize, encode the E4M3 payload.
template <bool kTabs, bool kExpand>
inline void group_compute(ScaleCodeTile &code) {
  BlockTile block;
  if constexpr (kExpand) {
    TEXPANDS(block, __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x3f80)));
  }

  BlockTile scaled;
  quantize<kTabs>(scaled, code, block);

  QuantBlockTile quant;
  TCVT(quant, scaled);
}

// SPMD entry, same external shape as mxquant::run; pointers are unused.
template <bool kTabs = false, bool kExpand = true>
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  (void)output;
  (void)scales;
  (void)input;

  const uint32_t tid = get_thread_idx();
  if (tid >= kPeCount) return;

  using WordTile = VecTileM32<uint32_t, kRows, 1>;

  for (int rb = 0; rb < kRowBlocks; ++rb) {
    ScaleCodeTile code0, code1, code2, code3, code4, code5, code6, code7;
    group_compute<kTabs, kExpand>(code0);
    group_compute<kTabs, kExpand>(code1);
    group_compute<kTabs, kExpand>(code2);
    group_compute<kTabs, kExpand>(code3);
    group_compute<kTabs, kExpand>(code4);
    group_compute<kTabs, kExpand>(code5);
    group_compute<kTabs, kExpand>(code6);
    group_compute<kTabs, kExpand>(code7);

    // Same scale pack as mxquant, minus the final GM stores.
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
    TPACK(low, p01, p23, 0x00000202);

    WordTile p45, p67, high;
    TPACK(p45, w4, w5, 0x00000101);
    TPACK(p67, w6, w7, 0x00000101);
    TPACK(high, p45, p67, 0x00000202);
  }
}

}  // namespace mxquant_compute
