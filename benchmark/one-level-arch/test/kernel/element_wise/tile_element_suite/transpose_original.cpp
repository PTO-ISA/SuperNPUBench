#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "basic_op/transpose/detail/transpose.hpp"
#include <cstdint>

#ifndef TRANSPOSE_PROFILE
#define TRANSPOSE_PROFILE 0
#endif
#ifndef TRANSPOSE_ALIAS_PROBE
#define TRANSPOSE_ALIAS_PROBE 0
#endif

namespace {
#if TRANSPOSE_PROFILE == 0
constexpr unsigned kRank = 2, kAxis0 = 0, kAxis1 = 1, kCount = 1476 * 32;
#elif TRANSPOSE_PROFILE == 1
constexpr unsigned kRank = 4, kAxis0 = 2, kAxis1 = 3, kCount = 8 * 4096 * 3;
#elif TRANSPOSE_PROFILE == 2
constexpr unsigned kRank = 4, kAxis0 = 1, kAxis1 = 2, kCount = 32 * 8 * 32;
#elif TRANSPOSE_PROFILE == 3
constexpr unsigned kRank = 5, kAxis0 = 1, kAxis1 = 3, kCount = 8 * 64 * 4 * 16 * 7;
#else
#error unsupported original transpose profile
#endif
constexpr bool kAlias = TRANSPOSE_ALIAS_PROBE != 0;
void set_bits(__half &value, uint16_t bits) {
  __builtin_memcpy(&value, &bits, sizeof(bits));
}
} // namespace

// 原 driver 的四种形状、完整容量和128项 shape 数组保持不变。
extern "C" {
alignas(64) __half transpose_input_guard_before[32];
alignas(64) __half transpose_input[kCount];
alignas(64) __half transpose_input_guard_after[32];
alignas(64) __half transpose_output_guard_before[32];
alignas(64) __half transpose_output[kCount];
alignas(64) __half transpose_output_guard_after[32];
#if TRANSPOSE_PROFILE == 0
uint32_t transpose_in_shape[128] = {1476, 32};
uint32_t transpose_out_shape[128] = {32, 1476};
#elif TRANSPOSE_PROFILE == 1
uint32_t transpose_in_shape[128] = {1, 8, 4096, 3};
uint32_t transpose_out_shape[128] = {1, 8, 3, 4096};
#elif TRANSPOSE_PROFILE == 2
uint32_t transpose_in_shape[128] = {1, 32, 8, 32};
uint32_t transpose_out_shape[128] = {1, 8, 32, 32};
#else
uint32_t transpose_in_shape[128] = {8, 64, 4, 16, 7};
uint32_t transpose_out_shape[128] = {8, 16, 4, 64, 7};
#endif
uint32_t transpose_status[8];
}

int main() {
  for (uint32_t i = 0; i < kCount; ++i) {
    set_bits(transpose_input[i], static_cast<uint16_t>(i * 40503u + 0x35a7u));
    set_bits(transpose_output[i], 0x5a5a);
  }
  for (uint32_t i = 0; i < 32; ++i) {
    set_bits(transpose_input_guard_before[i], 0x5a5a);
    set_bits(transpose_input_guard_after[i], 0x5a5a);
    set_bits(transpose_output_guard_before[i], 0x5a5a);
    set_bits(transpose_output_guard_after[i], 0x5a5a);
  }
  BENCHSTART;
  [[clang::noinline]]
  transpose<__half, 8, kCount, kCount, 512, kRank, kRank, kAxis0, kAxis1>(
      transpose_input, kAlias ? transpose_input : transpose_output,
      transpose_in_shape, transpose_out_shape);
  BENCHEND;
  transpose_status[0] = kCount;
  transpose_status[1] = 512;
  transpose_status[2] = kRank;
  transpose_status[3] = kAxis0;
  transpose_status[4] = kAxis1;
  transpose_status[5] = kAlias;
  transpose_status[6] = TRANSPOSE_PROFILE;
  transpose_status[7] = 0x54525031;
  return 0;
}
