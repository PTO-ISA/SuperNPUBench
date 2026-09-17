#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "basic_op/mxquant/mxquant.hpp"

// Input-materialisation-only variant: exactly the TEXPANDS ops that
// mxquant_compute uses to bring the input into the register file, and nothing
// else.  Built and measured so that
//
//     pure algorithm cost = compute_total - texpands_total
//
// i.e. the compute-only cycle count with the input TEXPANDS subtracted.  The
// fixed startup/BENCHSTART overhead is present in every build, so it cancels in
// the difference.  Each TEXPANDS is `asm volatile`, so the ops stay in the ELF
// even though their results are unused.  TIMING-ONLY build.

namespace mxquant_texpands {

using namespace pto;

using mxquant::BlockTile;

constexpr int kTotalRows = mxquant::kTotalRows;       // 512
constexpr int kRows = mxquant::kRows;                 // 32
constexpr int kCols = mxquant::kCols;                 // 256
constexpr int kBlock = mxquant::kBlock;               // 32
constexpr int kPeCount = mxquant::kPeCount;           // 4
constexpr int kRowsPerPe = mxquant::kRowsPerPe;       // 128
constexpr int kRowBlocks = mxquant::kRowBlocks;       // 4
constexpr int kBlocksPerRow = mxquant::kBlocksPerRow; // 8

// SPMD entry, same external shape as mxquant::run; pointers are unused.
inline void run(__fp8_e4m3 *output, __fp8_e8m0 *scales, const __bf16 *input) {
  (void)output;
  (void)scales;
  (void)input;

  const uint32_t tid = get_thread_idx();
  if (tid >= kPeCount) return;

  const __bf16 value =
      __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x3f80));

  for (int rb = 0; rb < kRowBlocks; ++rb) {
    BlockTile b0, b1, b2, b3, b4, b5, b6, b7;
    TEXPANDS(b0, value);
    TEXPANDS(b1, value);
    TEXPANDS(b2, value);
    TEXPANDS(b3, value);
    TEXPANDS(b4, value);
    TEXPANDS(b5, value);
    TEXPANDS(b6, value);
    TEXPANDS(b7, value);
  }
}

}  // namespace mxquant_texpands
