#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "basic_op/gather/detail/gather.hpp"
#include <cstdint>

#ifndef GATHER_PROFILE
#define GATHER_PROFILE 0
#endif
#ifndef GATHER_INDEX_ALIAS_PROBE
#define GATHER_INDEX_ALIAS_PROBE 0
#endif
#ifndef GATHER_ALIAS_PROBE
#define GATHER_ALIAS_PROBE 0
#endif
namespace {
#if GATHER_PROFILE == 0
constexpr uint32_t kK = 200000, kM = 202, kN = 97, kTileM = 128, kTileN = 128;
#elif GATHER_PROFILE == 1
constexpr uint32_t kK = 875000, kM = 8192, kN = 1, kTileM = 128, kTileN = 128;
#elif GATHER_PROFILE == 2
constexpr uint32_t kK = 754, kM = 640, kN = 17, kTileM = 128, kTileN = 128;
#elif GATHER_PROFILE == 3
constexpr uint32_t kK = 131072, kM = 32, kN = 256, kTileM = 32, kTileN = 64;
#else
#error unsupported original gather profile
#endif
constexpr bool kAlias = GATHER_ALIAS_PROBE != 0;
constexpr bool kIndexAlias = GATHER_INDEX_ALIAS_PROBE != 0;
constexpr uint32_t kInputCount = kK * kN;
constexpr uint32_t kOutputCount = kM * kN;
uint32_t row_index(uint32_t row) { return row % 17 == 0 ? kK - 1 : (row * 7u + 3) % 64; }
uint32_t input_bits(uint32_t index) {
  if constexpr (kIndexAlias)
    return (index * 13u + 5u) % kK;
  constexpr uint32_t edge[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
                               0x7fc12345u, 0x7f800123u, 1, 0x80000001u};
  return index % 32 < 8 ? edge[index % 32] : index * 104729u + 0x80003039u;
}
void set_bits(float &value, uint32_t bits) { __builtin_memcpy(&value, &bits, 4); }
} // namespace

// 保留原始容量、+11/+11/+17偏移；原driver的大栈分配仍保留在原文件。
// 此观测fixture的完整BSS表由ELF装载器清零，仅为索引初始引用的行填非零位值。
extern "C" {
alignas(64) float gather_input_guard_before[16];
alignas(64) float gather_input[kInputCount + 11];
alignas(4) float gather_input_guard_after[16];
// The cross-type alias fixture is compiled with -fno-strict-aliasing.
// Keep room for the entire original output when it shares the index backing.
alignas(64) uint32_t gather_indices[kIndexAlias ? kOutputCount + 17 : kM + 11];
alignas(64) float gather_output_guard_before[16];
alignas(64) float gather_output[kOutputCount + 17];
alignas(4) float gather_output_guard_after[16];
uint32_t gather_status[12];
}
int main() {
  if constexpr (kIndexAlias)
    for (uint32_t i = 0; i < kOutputCount + 17; ++i)
      gather_indices[i] = 0x5a5a5a5a;
  for (uint32_t i = 0; i < 11; ++i) {
    set_bits(gather_input[i], 0x5a5a5a5a);
    gather_indices[i] = 0x5a5a5a5a;
  }
  for (uint32_t row = 0; row < kM; ++row) {
    const uint32_t index = row_index(row);
    gather_indices[11 + row] = index;
    for (uint32_t column = 0; column < kN; ++column)
      set_bits(gather_input[11 + index * kN + column], input_bits(index * kN + column));
  }
  for (uint32_t i = 0; i < kOutputCount + 17; ++i)
    set_bits(gather_output[i], 0x5a5a5a5a);
  for (uint32_t i = 0; i < 16; ++i) {
    set_bits(gather_input_guard_before[i], 0x5a5a5a5a);
    set_bits(gather_input_guard_after[i], 0x5a5a5a5a);
    set_bits(gather_output_guard_before[i], 0x5a5a5a5a);
    set_bits(gather_output_guard_after[i], 0x5a5a5a5a);
  }
  BENCHSTART;
  [[clang::noinline]]
  gather<float, uint32_t, kK, kM, kN, kTileM, kTileN>(
      gather_input + 11, gather_indices + 11,
      kIndexAlias ? reinterpret_cast<float *>(gather_indices + 17)
                  : (kAlias ? gather_input + 17 : gather_output + 17));
  BENCHEND;
  gather_status[0] = kK; gather_status[1] = kM; gather_status[2] = kN;
  gather_status[3] = kTileM; gather_status[4] = kTileN;
  gather_status[5] = 11; gather_status[6] = 11; gather_status[7] = 17;
  gather_status[8] = kIndexAlias ? 2 : kAlias; gather_status[9] = GATHER_PROFILE;
  gather_status[10] = kInputCount; gather_status[11] = 0x47415431;
  return 0;
}
