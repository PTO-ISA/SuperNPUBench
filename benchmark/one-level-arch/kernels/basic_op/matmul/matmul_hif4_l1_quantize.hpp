#pragma once

// Fused 4-PE matmul followed by level-1 HiF4 quantization.
//
// Mathematical result:
//   C = A * B, C is accumulated in FP32.
//
// Quantization is performed independently for every row and every 64
// consecutive C columns.  This operator implements only the first HiF4
// scale level:
//
//   amax     = max(abs(C[row, n:n+64]))
//   base_hp  = amax / 1.75
//   base     = E6M2_RNE(base_hp)
//   data     = S1P2_RNE_SAT(C / decode(base))
//
// E1_8 and E1_16 are intentionally absent.  Consequently the normalization
// target is S1P2's own maximum 1.75 (= 7/4), not the full three-level HiF4
// maximum 7.  The output scale matrix contains one raw E6M2 byte per
// [row, 64-column group].  It can later become bits [7:0] of the full U32
// HiF4 scale word, with bits [31:8] initially zero.
//
// The two [32,32] scaled halves are joined with the TileOP high-level assembly
// API:
//
//   TileArray<BfHalfTile, 1, 2> -> TCVT(slot, half) -> TASSEMBLY<BfBlockTile>
//
// This gives the final conversion one contiguous [32,64] source. TASSEMBLY
// lowers the two region producers to the B.ASSEMBLE INIT/LAST lifecycle.

#include "basic_op/matmul/matmul_shared.hpp"
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

namespace matmul_hif4_l1_detail {

using namespace pto;

constexpr int kPeNum = 4;
constexpr int kRows = 32;
constexpr int kGroup = 64;
constexpr int kHalf = 32;

using Fp32HalfTile =
    Tile<Location::Vec, float, kRows, kHalf, BLayout::RowMajor>;
using BfHalfTile =
    Tile<Location::Vec, __bf16, kRows, kHalf, BLayout::RowMajor>;
using BfBlockTile =
    Tile<Location::Vec, __bf16, kRows, kGroup, BLayout::RowMajor>;
using BfRowTile =
    Tile<Location::Vec, __bf16, kRows, 1, BLayout::RowMajor>;
using U16RowTile =
    Tile<Location::Vec, uint16_t, kRows, 1, BLayout::RowMajor>;

// A byte carrier for one valid E6M2 code per row. Physical Cols=2 is needed
// because the Tile minimum capacity is 128 B: it makes the capacity-derived
// row count equal to BfRowTile's while ValidCol remains one.
using E6M2CodeTile =
    Tile<Location::Vec, uint8_t, kRows, 2, BLayout::RowMajor, kRows, 1>;

// HiF4X2 contains two logical S1P2 lanes in one byte carrier.
using Hif4BlockTile =
    Tile<Location::Vec, __fp4_hif4x2, kRows, kGroup, BLayout::RowMajor>;

static_assert(BfHalfTile::LogicalTileBytes * 2 ==
                  BfBlockTile::LogicalTileBytes,
              "TASSEMBLY requires the two fragment carriers to cover the "
              "whole parent carrier");
static_assert(BfRowTile::LogicalTileBytes == E6M2CodeTile::LogicalTileBytes,
              "BF16/E6M2 conversion carriers must have equal capacity");

// BF16 positive scale -> E6M2 RNE code, and reconstruct the exact rounded
// E6M2 value as BF16.  Current TileOP headers expose the scalar E6M2 C++ type
// but neither the public type-code table nor the assembler accepts it in a
// B.DATR conversion.  Encoding through integer Tile operations keeps this
// benchmark independent of that missing conversion surface.
//
// BF16: exponent bias 127, 7 fraction bits.
// E6M2: exponent bias 48,  2 fraction bits.
// RNE from 7 to 2 fraction bits is: bits + 0x0f + kept_lsb, then drop 5 bits.
inline void encode_e6m2(E6M2CodeTile &dst, BfRowTile &rounded_base,
                        BfRowTile &src) {
  auto src_view = reinterpret_tile<uint16_t>(src);
  U16RowTile raw;
  TCVT(raw, src_view);

  U16RowTile kept_lsb;
  TSHRS(kept_lsb, raw, static_cast<uint16_t>(5));
  TANDS(kept_lsb, kept_lsb, static_cast<uint16_t>(1));

  U16RowTile rounded;
  TADDS(rounded, raw, static_cast<uint16_t>(0x000f));
  TADD(rounded, rounded, kept_lsb);

  U16RowTile exponent;
  TSHRS(exponent, rounded, static_cast<uint16_t>(7));
  TANDS(exponent, exponent, static_cast<uint16_t>(0x00ff));

  U16RowTile e6_exponent;
  TSUBS(e6_exponent, exponent, static_cast<uint16_t>(79));

  U16RowTile mantissa;
  TSHRS(mantissa, rounded, static_cast<uint16_t>(5));
  TANDS(mantissa, mantissa, static_cast<uint16_t>(3));

  U16RowTile code;
  TSHLS(code, e6_exponent, static_cast<uint16_t>(2));
  TOR(code, code, mantissa);
  TMINS(code, code, static_cast<uint16_t>(0x00fe));

  // E6M2 has no zero/subnormal. Values below 2^-48, including an all-zero
  // row, use the minimum finite code 0x00. Inf/NaN use E6M2 qNaN 0xff.
  U16RowTile underflow;
  U16RowTile special;
  TCMPS<CmpMode::LT>(underflow, exponent, static_cast<uint16_t>(79));
  TCMPS<CmpMode::EQ>(special, exponent, static_cast<uint16_t>(255));
  U16RowTile constant;
  TEXPANDS(constant, static_cast<uint16_t>(0));
  TSEL(code, underflow, constant);
  TEXPANDS(constant, static_cast<uint16_t>(0x00ff));
  TSEL(code, special, constant);

  // Compact scale output: one byte per row/group.
  TCVT(dst, code);

  // Decode finite E6M2 code back into exact BF16 bits:
  // bf_exp = e6_exp - 48 + 127 = e6_exp + 79,
  // bf_mantissa[6:5] = e6_mantissa[1:0].
  U16RowTile base_exp;
  TSHRS(base_exp, code, static_cast<uint16_t>(2));
  TADDS(base_exp, base_exp, static_cast<uint16_t>(79));
  TSHLS(base_exp, base_exp, static_cast<uint16_t>(7));

  U16RowTile base_mantissa;
  TANDS(base_mantissa, code, static_cast<uint16_t>(3));
  TSHLS(base_mantissa, base_mantissa, static_cast<uint16_t>(5));

  U16RowTile base_bits;
  TOR(base_bits, base_exp, base_mantissa);
  TEXPANDS(constant, static_cast<uint16_t>(0x7fc0));
  TSEL(base_bits, special, constant);

  auto base_view = reinterpret_tile<__bf16>(base_bits);
  TCVT(rounded_base, base_view);
}

// Quantize one [32,64] output block.  Row maxima are reduced on the two
// [32,32] halves separately so each TROWMAX source remains 2 KiB, then merged
// elementwise.  A single row scale is applied to both halves.
inline void quantize_block(BfHalfTile &scaled_left,
                           BfHalfTile &scaled_right,
                           E6M2CodeTile &scale, BfHalfTile &left,
                           BfHalfTile &right) {
  BfHalfTile abs_left;
  BfHalfTile abs_right;
  TABS(abs_left, left);
  TABS(abs_right, right);

  BfRowTile max_left;
  BfRowTile max_right;
  BfRowTile amax;
  TROWMAX(max_left, abs_left);
  TROWMAX(max_right, abs_right);
  TMAX(amax, max_left, max_right);

  // First-level-only HiF4: base target is amax / S1P2_max = amax * 4/7.
  BfRowTile base_hp;
  TMULS(base_hp, amax, static_cast<__bf16>(4.0f / 7.0f));

  BfRowTile base;
  BfRowTile inv_base;
  encode_e6m2(scale, base, base_hp);
  TRECIP(inv_base, base);

  TROWEXPANDMUL(scaled_left, left, inv_base);
  TROWEXPANDMUL(scaled_right, right, inv_base);
}

} // namespace matmul_hif4_l1_detail

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_hif4_l1_quantize(__fp4_hif4x2 *data_ptr, uint8_t *scale_ptr,
                             float *c_scratch, dtype *a_ptr, dtype *b_ptr) {
  using namespace matmul_hif4_l1_detail;

  constexpr int kGroupM = tM <= 128 ? tM : 128;
  constexpr int kTileRows = kGroupM <= 64 ? 64 : 128;
  constexpr int kValidRowM = (gM < kTileRows) ? gM : kTileRows;
  constexpr int kPeM = kValidRowM <= 64 ? 16 : 32;
  constexpr int kMb = (gM + kGroupM - 1) / kGroupM;
  constexpr int kScaleCols = gN / kGroup;

  static_assert(kPeM == kRows,
                "the HiF4 quantizer processes 32 result rows per PE; use "
                "tM>=128");
  static_assert(gN % kGroup == 0,
                "HiF4 level-1 quantization requires N to be a multiple of 64");
  static_assert(gM % kGroupM == 0 || gM < kGroupM,
                "gM must satisfy matmul_shared's group-M partition");

  // Phase 1: cooperative 4-PE matmul materializes FP32 C.
  matmul_shared<dtype, gM, gN, gK, tM, tN, tK>(c_scratch, a_ptr, b_ptr);

  using gmC = global_tensor<float, RowMajor<gM, gN>>;
  using gmData = global_tensor<__fp4_hif4x2, RowMajor<gM, gN>>;
  using gmScale = global_tensor<uint8_t, RowMajor<gM, kScaleCols>>;

  const uint32_t tid = get_thread_idx();

#pragma clang loop unroll(full)
  for (int mb = 0; mb < kMb; ++mb) {
    const int row0 =
        (mb * kPeNum + static_cast<int>(tid)) * kPeM;
    if constexpr (gM < kGroupM) {
      if (row0 >= gM)
        continue;
    }

#pragma clang loop unroll(full)
    for (int ng = 0; ng < kScaleCols; ++ng) {
      const int col0 = ng * kGroup;

      global_iterator<gmC, Fp32HalfTile> left_iter(
          c_scratch + row0 * gN + col0);
      global_iterator<gmC, Fp32HalfTile> right_iter(
          c_scratch + row0 * gN + col0 + kHalf);

      auto g_left = left_iter(0, 0);
      auto g_right = right_iter(0, 0);
      Fp32HalfTile c_left;
      Fp32HalfTile c_right;
      TLOAD(c_left, g_left);
      TLOAD(c_right, g_right);

      BfHalfTile bf_left;
      BfHalfTile bf_right;
      TCVT(bf_left, c_left);
      TCVT(bf_right, c_right);

      BfHalfTile scaled_left;
      BfHalfTile scaled_right;
      E6M2CodeTile scale;
      quantize_block(scaled_left, scaled_right, scale, bf_left, bf_right);

      // HiF4X2 packs two logical columns in one byte. Keep logical tensor
      // dimensions in gmData, but advance the physical base in bytes.
      auto *data_base = reinterpret_cast<__fp4_hif4x2 *>(
          reinterpret_cast<uint8_t *>(data_ptr) + row0 * (gN / 2) +
          col0 / 2);
      TileArray<BfHalfTile, 1, 2> parts;
      TCVT(parts[0][0], scaled_left);
      TCVT(parts[0][1], scaled_right);
      BfBlockTile scaled = TASSEMBLY<BfBlockTile>(std::move(parts));

      // BF16 -> packed sign-magnitude S1P2, default/RNE rounding with
      // saturation supplied by the HiF4X2 conversion contract.
      Hif4BlockTile data;
      TCVT(data, scaled);
      global_iterator<gmData, Hif4BlockTile> data_iter(data_base);
      auto g_data = data_iter(0, 0);
      TSTORE(g_data, data);

      global_iterator<gmScale, E6M2CodeTile> scale_iter(
          scale_ptr + row0 * kScaleCols + ng);
      auto g_scale = scale_iter(0, 0);
      TSTORE(g_scale, scale);
    }
  }
}
