#pragma once

// Shared scalar reference for all mxquant test entries (64-wide and
// 1024-wide variants).  Kept deliberately separate from the TileOP kernels
// so a byte-for-byte check is possible without hiding the kernel dataflow.
// Compiled only into res_check=on builds.

#include <common/pto_tileop.hpp>
#include <cstdint>

#include "linx_print.h"

namespace mxquant_ref {

inline uint8_t round_shift_even(uint32_t value, int shift) {
  if (shift <= 0) return static_cast<uint8_t>(value << (-shift));

  const uint32_t base = value >> shift;
  const uint32_t remainder = value & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  const bool round_up = remainder > half ||
                        (remainder == half && (base & 1u) != 0);
  return static_cast<uint8_t>(base + round_up);
}

inline uint8_t encode_e4m3(uint16_t bf16_bits) {
  const uint8_t sign = static_cast<uint8_t>((bf16_bits >> 8) & 0x80);
  const uint16_t abs_bits = bf16_bits & 0x7fff;
  const int bf_exp = (abs_bits >> 7) & 0xff;
  const uint32_t significand = 128u + (abs_bits & 0x7f);

  if (bf_exp == 0) return sign;

  const int exponent = bf_exp - 127;
  if (exponent > 8) return static_cast<uint8_t>(sign | 0x7e);
  if (exponent < -9) return sign;

  if (exponent < -6) {
    const int shift = -(exponent + 2);
    const uint8_t mantissa = round_shift_even(significand, shift);
    return mantissa >= 8 ? static_cast<uint8_t>(sign | 0x08)
                         : static_cast<uint8_t>(sign | mantissa);
  }

  int mantissa = static_cast<int>(round_shift_even(abs_bits & 0x7f, 4));
  int biased_exponent = exponent + 7;
  if (mantissa >= 8) {
    mantissa = 0;
    ++biased_exponent;
  }
  // E4M3 max normal is 448 (0x7E); OCP MX 6.3 clamps overflow to +-448
  // instead of emitting the NaN pattern 0x7F (exponent 8, mantissa 7 = 480).
  if (biased_exponent > 15 || (biased_exponent == 15 && mantissa > 6)) {
    return static_cast<uint8_t>(sign | 0x7e);
  }
  return static_cast<uint8_t>(sign | (biased_exponent << 3) | mantissa);
}

inline uint8_t encode_e8m0(uint16_t amax_bits) {
  const uint16_t abs_bits = amax_bits & 0x7fff;
  if (abs_bits == 0) return 0;

  const int bf_exp = (abs_bits >> 7) & 0xff;
  // OCP MX section 6.3 / AMD Quark: X = 2^(floor(log2(amax)) - 8), where 8 is
  // the max exponent of E4M3 (448 = 1.75*2^8; the largest power-of-two
  // element is 256).  Scaled values in (448, 512) saturate in encode_e4m3.
  int exponent = (bf_exp - 127) - 8;
  if (exponent < -127) exponent = -127;
  if (exponent > 127) exponent = 127;
  return static_cast<uint8_t>(exponent + 127);
}

inline void golden(const __bf16 *input, uint8_t *output, uint8_t *scales,
                   int rows, int cols, int block) {
  const int blocks_per_row = cols / block;
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      uint16_t amax = 0;
      for (int col = 0; col < block; ++col) {
        const int index = row * cols + blk * block + col;
        const uint16_t bits = *reinterpret_cast<const uint16_t *>(&input[index]);
        amax = (bits & 0x7fff) > amax ? (bits & 0x7fff) : amax;
      }

      const uint8_t scale = encode_e8m0(amax);
      scales[row * blocks_per_row + blk] = scale;
      const int scale_exponent = static_cast<int>(scale) - 127;

      for (int col = 0; col < block; ++col) {
        const int index = row * cols + blk * block + col;
        const uint16_t bits =
            *reinterpret_cast<const uint16_t *>(&input[index]);
        const uint16_t sign = bits & 0x8000;
        const uint16_t abs_bits = bits & 0x7fff;
        int exponent = (abs_bits >> 7) & 0xff;
        if (exponent != 0) {
          exponent += -scale_exponent;
          if (exponent < 1) exponent = 0;
          if (exponent > 254) exponent = 255;
        }
        const uint16_t scaled =
            static_cast<uint16_t>(sign | (exponent << 7) | (abs_bits & 0x7f));
        output[index] = encode_e4m3(scaled);
      }
    }
  }
}

// Note: there is deliberately no device-side input generator here.  A scalar
// fill loop writing this tensor was measured not to become visible to the
// buffer the kernel and the syscall path read, which silently degraded the
// test to an all-zero input -- where the golden is also all zero and every
// comparison passes vacuously.  The host owns the input (input.bin); the
// matching distribution lives in run_mxquant_check.py.

inline int check_result(uint32_t tid, const __fp8_e4m3 *output,
                        const __fp8_e8m0 *scales, const uint8_t *golden_output,
                        const uint8_t *golden_scales, int rows, int cols,
                        int blocks_per_row) {
  int failures = 0;
  for (int i = 0; i < rows * blocks_per_row; ++i) {
    const uint8_t got = __fp8_e8m0_STORAGE(scales[i]);
    if (got != golden_scales[i]) {
      if (failures < 8) {
        linxi_put("scale mismatch ");
        linxi_put_i64(i);
        linxi_put(" got ");
        linxi_put_hex(got);
        linxi_put(" want ");
        linxi_put_hex(golden_scales[i]);
        linxi_putc('\n');
      }
      ++failures;
    }
  }

  for (int i = 0; i < rows * cols; ++i) {
    const uint8_t got = __fp8_e4m3_STORAGE(output[i]);
    if (got != golden_output[i]) {
      if (failures < 8) {
        linxi_put("payload mismatch ");
        linxi_put_i64(i);
        linxi_put(" got ");
        linxi_put_hex(got);
        linxi_put(" want ");
        linxi_put_hex(golden_output[i]);
        linxi_putc('\n');
      }
      ++failures;
    }
  }

  linxi_put("PE ");
  linxi_put_i64(tid);
  linxi_put(" MXQUANT ");
  linxi_puts(failures == 0 ? "PASS" : "FAIL");
  linxi_put_kv("fails", failures);
  return failures;
}

}  // namespace mxquant_ref
