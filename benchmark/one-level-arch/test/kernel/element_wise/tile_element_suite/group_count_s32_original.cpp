#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                        \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "group_count_s32_original requires the official typed element API"
#endif

#include "benchmark.h"
#include "linx_print.h"
#include "basic_op/group_count/group_count_aux_fi.hpp"

#include <cstddef>
#include <cstdint>

#ifndef GROUP_COUNT_S32_STRONG_CASE
#define GROUP_COUNT_S32_STRONG_CASE 0
#endif

namespace group_count_s32_original_benchmark {

using supernpu::tile_isa::group_count;

constexpr std::size_t kInputElements = 256;
constexpr std::size_t kUsedElements = 128;
constexpr std::size_t kOutputElements = 64;
constexpr std::size_t kBins = 32;
constexpr int32_t kUnusedOutput = INT32_C(0x5a5a6b6b);
constexpr uint32_t kMagic = UINT32_C(0x47435332); // GCS2

static void run_group_count(int32_t *input, int32_t *output) {
  // 保留真实模板调用；程序不声明 layout、lane、mask 或中间 Tile。
  [[clang::noinline]] group_count<16, 8, 32>(input, output);
}

static int32_t seeded_index(uint32_t &state, std::size_t element) {
  state = state * UINT32_C(1664525) + UINT32_C(1013904223);
  int32_t value = static_cast<int32_t>((state >> 16) % kBins);
  if (element == 0)
    value = -1;
  else if (element == 1)
    value = 32;
  else if (element == 2)
    value = 99;
  else if (element == 63)
    value = -7;
  else if (element == 127)
    value = INT32_MAX;
  return value;
}

} // namespace group_count_s32_original_benchmark

using namespace group_count_s32_original_benchmark;

#if GROUP_COUNT_S32_STRONG_CASE
extern "C" {
alignas(4096) int32_t group_count_s32_strong_input[kInputElements];
alignas(4096) int32_t group_count_s32_strong_output[kOutputElements];
alignas(4096) int32_t group_count_s32_strong_alias[kInputElements];
alignas(32) uint32_t group_count_s32_strong_status[12];
}

int main() {
  uint32_t state = 1;
  for (std::size_t element = 0; element < kInputElements; ++element) {
    const int32_t value =
        element < kUsedElements
            ? seeded_index(state, element)
            : static_cast<int32_t>(UINT32_C(0x6b6b0000) + element);
    group_count_s32_strong_input[element] = value;
    group_count_s32_strong_alias[element] = value;
  }
  for (std::size_t element = 0; element < kOutputElements; ++element)
    group_count_s32_strong_output[element] = kUnusedOutput;

  // 第二次调用必须在 kernel 内重新清零 32 个 bin，结果不能翻倍。
  BENCHSTART;
  run_group_count(group_count_s32_strong_input, group_count_s32_strong_output);
  run_group_count(group_count_s32_strong_input, group_count_s32_strong_output);
  // out 与 input 完全重叠，验证“先快照全部 128 个下标，再清输出”。
  run_group_count(group_count_s32_strong_alias, group_count_s32_strong_alias);
  BENCHEND;

  int32_t expected[kBins] = {};
  uint32_t valid = 0;
  uint32_t ignored = 0;
  for (std::size_t element = 0; element < kUsedElements; ++element) {
    const int32_t index = group_count_s32_strong_input[element];
    if (index >= 0 && index < static_cast<int32_t>(kBins)) {
      ++expected[index];
      ++valid;
    } else {
      ++ignored;
    }
  }

  uint32_t output_failures = 0;
  uint32_t output_tail_failures = 0;
  uint32_t alias_failures = 0;
  uint32_t alias_tail_failures = 0;
  uint32_t checksum = 0;
  for (std::size_t bin = 0; bin < kBins; ++bin) {
    output_failures += group_count_s32_strong_output[bin] != expected[bin];
    alias_failures += group_count_s32_strong_alias[bin] != expected[bin];
    checksum += static_cast<uint32_t>(expected[bin]);
  }
  for (std::size_t element = kBins; element < kOutputElements; ++element)
    output_tail_failures +=
        group_count_s32_strong_output[element] != kUnusedOutput;
  for (std::size_t element = kBins; element < kInputElements; ++element)
    alias_tail_failures += group_count_s32_strong_alias[element] !=
                           group_count_s32_strong_input[element];

  const uint32_t failures = output_failures + output_tail_failures +
                            alias_failures + alias_tail_failures;
  group_count_s32_strong_status[0] = failures;
  group_count_s32_strong_status[1] = output_failures;
  group_count_s32_strong_status[2] = output_tail_failures;
  group_count_s32_strong_status[3] = alias_failures;
  group_count_s32_strong_status[4] = alias_tail_failures;
  group_count_s32_strong_status[5] = kUsedElements;
  group_count_s32_strong_status[6] = kBins;
  group_count_s32_strong_status[7] = 2;
  group_count_s32_strong_status[8] = valid;
  group_count_s32_strong_status[9] = ignored;
  group_count_s32_strong_status[10] = checksum;
  group_count_s32_strong_status[11] = kMagic;

  linxi_puts("=== group_count_s32_strong ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
#else
extern "C" {
alignas(4096) int32_t group_count_s32_original_input[kInputElements] = {};
alignas(4096) int32_t group_count_s32_original_output[kOutputElements] = {};
alignas(32) uint32_t group_count_s32_original_status[12];
}

int main() {
  BENCHSTART;
  run_group_count(group_count_s32_original_input,
                  group_count_s32_original_output);
  BENCHEND;

  uint32_t input_failures = 0;
  uint32_t output_failures = 0;
  uint32_t output_tail_failures = 0;
  for (std::size_t element = 0; element < kInputElements; ++element)
    input_failures += group_count_s32_original_input[element] != 0;
  output_failures += group_count_s32_original_output[0] != 128;
  for (std::size_t bin = 1; bin < kBins; ++bin)
    output_failures += group_count_s32_original_output[bin] != 0;
  for (std::size_t element = kBins; element < kOutputElements; ++element)
    output_tail_failures += group_count_s32_original_output[element] != 0;

  const uint32_t failures =
      input_failures + output_failures + output_tail_failures;
  group_count_s32_original_status[0] = failures;
  group_count_s32_original_status[1] = input_failures;
  group_count_s32_original_status[2] = output_failures;
  group_count_s32_original_status[3] = output_tail_failures;
  group_count_s32_original_status[4] = kInputElements;
  group_count_s32_original_status[5] = kUsedElements;
  group_count_s32_original_status[6] = kOutputElements;
  group_count_s32_original_status[7] = kBins;
  group_count_s32_original_status[8] = 1;
  group_count_s32_original_status[9] = 128;
  group_count_s32_original_status[10] = 0;
  group_count_s32_original_status[11] = kMagic;

  linxi_puts("=== group_count_s32_original ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
#endif
