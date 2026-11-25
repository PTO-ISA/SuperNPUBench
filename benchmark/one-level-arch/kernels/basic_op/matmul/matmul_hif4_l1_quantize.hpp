#pragma once

// 4-PE matmul followed by level-1 HiF4 quantization.
//
// Mathematical result:
//   C = A * B^T, B is [N,K], C is accumulated in FP32.
//
// Quantization is performed independently for every row and every 64
// consecutive C columns.  This operator implements only the first HiF4
// scale level:
//
//   c        = BF16_RNE(C[row, n:n+64])
//   amax     = max(abs(c))
//   base_hp  = BF16_RNE(amax * BF16_RNE(4/7))
//   base     = E6M2_RNE(base_hp)
//   inv_base = BF16_RNE(1 / decode(base))
//   data     = S1P2_RNE_SAT(BF16_RNE(c * inv_base))
//
// E1_8 and E1_16 are intentionally absent.  Consequently the normalization
// target is S1P2's own maximum 1.75 (= 7/4), not the full three-level HiF4
// maximum 7.  The output scale matrix contains one raw E6M2 byte per
// [row, 64-column group].  It can later become bits [7:0] of the full U32
// HiF4 scale word, with bits [31:8] initially zero.
//
// HiF4 has no standalone TCVT encoding. This correctness baseline keeps
// the cooperative matmul and performs explicit scalar BF16/E6M2/S1P2
// quantization. It does not claim vector-quantization performance.
#include "basic_op/matmul/matmul_shared.hpp"
#include "basic_op/matmul/hif4_s1p2.hpp"

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_hif4_l1_quantize(__fp4_hif4x2 *data_ptr, uint8_t *scale_ptr,
                            float *c_scratch, dtype *a_ptr, dtype *b_ptr) {
  constexpr int groupM = tM <= 128 ? tM : 128;
  static_assert(groupM == 128 && gM % 128 == 0,
                "HiF4 L1 requires complete 128-row groups (32 rows per PE)");
  static_assert(gN % 64 == 0, "HiF4 L1 requires N divisible by 64");
  matmul_shared<dtype, gM, gN, gK, tM, tN, tK>(c_scratch, a_ptr, b_ptr);
  auto *bytes = reinterpret_cast<uint8_t *>(data_ptr);
  const int pe = get_thread_idx();
  for (int mb = 0; mb < gM / 128; ++mb) {
    for (int r = 0; r < 32; ++r) {
      const int row = mb * 128 + pe * 32 + r;
      for (int ng = 0; ng < gN / 64; ++ng) {
        uint16_t values[64];
        uint16_t amax = 0;
        bool special = false;
        for (int n = 0; n < 64; ++n) {
          values[n] = hif4::bf16_bits(c_scratch[row * gN + ng * 64 + n]);
          const uint16_t magnitude = values[n] & 0x7fff;
          special |= magnitude >= 0x7f80;
          if (magnitude > amax) amax = magnitude;
        }
        const auto base_hp = hif4::bf16_bits(
            hif4::bf16_value(amax) * hif4::bf16_value(hif4::bf16_bits(4.0f / 7.0f)));
        const uint8_t scale = special ? 255 : hif4::e6m2_from_bf16(base_hp);
        scale_ptr[row * (gN / 64) + ng] = scale;
        const uint16_t base_bits = scale == 255 ? 0x7fc0 :
            static_cast<uint16_t>(((scale / 4 + 79) << 7) | ((scale % 4) << 5));
        const float reciprocal = hif4::bf16_value(
            hif4::bf16_bits(1.0f / hif4::bf16_value(base_bits)));
        for (int n = 0; n < 64; n += 2) {
          const auto lo = hif4::s1p2_from_bf16(hif4::bf16_bits(hif4::bf16_value(values[n]) * reciprocal));
          const auto hi = hif4::s1p2_from_bf16(hif4::bf16_bits(hif4::bf16_value(values[n+1]) * reciprocal));
          bytes[row * (gN / 2) + ng * 32 + n / 2] = lo | (hi << 4);
        }
      }
    }
  }
}
