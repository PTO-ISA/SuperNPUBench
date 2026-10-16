#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace generic_predicated_cfg_i32_benchmark {

constexpr unsigned kElements = 32;
constexpr unsigned kValidElements = 29;
constexpr unsigned kGuardElements = 8;
constexpr int32_t kGuardValue = INT32_C(0x5a5a6b6b);

__attribute__((noinline)) void nested_three_way(
    const int32_t *__restrict a, const int32_t *__restrict control,
    const int32_t *__restrict divisor, int32_t *__restrict output,
    unsigned valid, bool enabled) {
#pragma pto element for
  for (unsigned element = 0; element < kElements; ++element) {
    if (enabled && element < valid) {
      const int32_t value = a[element];
      const int32_t selector = control[element];
      int32_t result;
      if (selector < 0) {
        result = (value + 21) / divisor[element];
      } else {
        if (selector == 0)
          result = value * 3 - 17;
        else
          result = (value ^ INT32_C(0x13579bdf)) + 9;
      }
      output[element] = result;
    }
  }
}

__attribute__((noinline)) void reversed_continue(
    const int32_t *__restrict a, const int32_t *__restrict control,
    const int32_t *__restrict divisor, int32_t *__restrict output,
    unsigned valid, bool enabled) {
#pragma pto element for
  for (unsigned element = 0; element < kElements; ++element) {
    if (!enabled || element >= valid)
      continue;

    const int32_t value = a[element];
    const int32_t selector = control[element];
    int32_t result;
    if (selector > 0)
      result = (value ^ INT32_C(0x13579bdf)) + 9;
    else if (selector == 0)
      result = value * 3 - 17;
    else
      result = (value + 21) / divisor[element];
    output[element] = result;
  }
}

__attribute__((noinline)) void direct_signed_store(
    const int32_t *__restrict a, const int32_t *__restrict control,
    const int32_t *__restrict divisor, int32_t *__restrict output,
    unsigned valid, bool enabled) {
#pragma pto element for
  for (unsigned element = 0; element < kElements; ++element) {
    if (enabled && element < valid && control[element] < 0)
      output[element] = (a[element] + 21) / divisor[element];
  }
}

} // namespace generic_predicated_cfg_i32_benchmark

using namespace generic_predicated_cfg_i32_benchmark;

extern "C" {
alignas(4096) int32_t generic_predicated_cfg_i32_input[kElements];
alignas(4096) int32_t generic_predicated_cfg_i32_control[kElements];
alignas(4096) int32_t generic_predicated_cfg_i32_divisor[kElements];
alignas(4096) int32_t
    generic_predicated_cfg_i32_nested[kElements + 2 * kGuardElements];
alignas(4096) int32_t
    generic_predicated_cfg_i32_reversed[kElements + 2 * kGuardElements];
alignas(4096) int32_t
    generic_predicated_cfg_i32_direct[kElements + 2 * kGuardElements];
alignas(32) uint32_t generic_predicated_cfg_i32_status[16];
}

int main() {
  for (unsigned element = 0; element < kElements; ++element) {
    int32_t value = static_cast<int32_t>(element * 7919u + 12345u) - 131071;
    if (element == 0)
      value = -100;
    if (element == 3)
      value = 100;
    generic_predicated_cfg_i32_input[element] = value;

    const int32_t selector = static_cast<int32_t>(element % 3u) - 1;
    generic_predicated_cfg_i32_control[element] = selector;
    generic_predicated_cfg_i32_divisor[element] =
        selector < 0 ? ((element & 1u) != 0u ? -5 : 3) : 0;
  }
  for (unsigned element = 0; element < kElements + 2 * kGuardElements;
       ++element) {
    generic_predicated_cfg_i32_nested[element] = kGuardValue;
    generic_predicated_cfg_i32_reversed[element] = kGuardValue;
    generic_predicated_cfg_i32_direct[element] = kGuardValue;
  }

  // 全 inactive 时所有 GM 指针均无效。正确的 mask 必须在任何 load、除法或
  // store 之前生效，因此这个调用只验证无探测、无故障和自然返回。
  nested_three_way(nullptr, nullptr, nullptr, nullptr, kElements, false);
  reversed_continue(nullptr, nullptr, nullptr, nullptr, kElements, false);
  direct_signed_store(nullptr, nullptr, nullptr, nullptr, kElements, false);

  BENCHSTART;
  nested_three_way(
      generic_predicated_cfg_i32_input,
      generic_predicated_cfg_i32_control,
      generic_predicated_cfg_i32_divisor,
      generic_predicated_cfg_i32_nested + kGuardElements,
      kValidElements, true);
  reversed_continue(
      generic_predicated_cfg_i32_input,
      generic_predicated_cfg_i32_control,
      generic_predicated_cfg_i32_divisor,
      generic_predicated_cfg_i32_reversed + kGuardElements,
      kValidElements, true);
  direct_signed_store(
      generic_predicated_cfg_i32_input,
      generic_predicated_cfg_i32_control,
      generic_predicated_cfg_i32_divisor,
      generic_predicated_cfg_i32_direct + kGuardElements,
      kValidElements, true);
  BENCHEND;

  uint32_t failures = 0;
  uint32_t guard_failures = 0;
  uint32_t nested_checksum = 0;
  uint32_t reversed_checksum = 0;
  uint32_t direct_checksum = 0;
  uint32_t direct_active = 0;
  for (unsigned element = 0; element < kElements; ++element) {
    const int32_t value = generic_predicated_cfg_i32_input[element];
    const int32_t selector = generic_predicated_cfg_i32_control[element];
    int32_t expected = kGuardValue;
    if (element < kValidElements) {
      if (selector < 0)
        expected = (value + 21) /
                   generic_predicated_cfg_i32_divisor[element];
      else if (selector == 0)
        expected = value * 3 - 17;
      else
        expected = (value ^ INT32_C(0x13579bdf)) + 9;
    }
    const int32_t nested =
        generic_predicated_cfg_i32_nested[kGuardElements + element];
    const int32_t reversed =
        generic_predicated_cfg_i32_reversed[kGuardElements + element];
    int32_t direct_expected = kGuardValue;
    if (element < kValidElements && selector < 0) {
      direct_expected =
          (value + 21) / generic_predicated_cfg_i32_divisor[element];
      ++direct_active;
    }
    const int32_t direct =
        generic_predicated_cfg_i32_direct[kGuardElements + element];
    failures += nested != expected;
    failures += reversed != expected;
    failures += direct != direct_expected;
    nested_checksum += static_cast<uint32_t>(nested);
    reversed_checksum += static_cast<uint32_t>(reversed);
    direct_checksum += static_cast<uint32_t>(direct);
  }
  for (unsigned element = 0; element < kGuardElements; ++element) {
    const unsigned suffix = kGuardElements + kElements + element;
    guard_failures += generic_predicated_cfg_i32_nested[element] != kGuardValue;
    guard_failures += generic_predicated_cfg_i32_nested[suffix] != kGuardValue;
    guard_failures +=
        generic_predicated_cfg_i32_reversed[element] != kGuardValue;
    guard_failures +=
        generic_predicated_cfg_i32_reversed[suffix] != kGuardValue;
    guard_failures += generic_predicated_cfg_i32_direct[element] != kGuardValue;
    guard_failures += generic_predicated_cfg_i32_direct[suffix] != kGuardValue;
  }
  failures += guard_failures;

  generic_predicated_cfg_i32_status[0] = kElements;
  generic_predicated_cfg_i32_status[1] = kValidElements;
  generic_predicated_cfg_i32_status[2] = failures;
  generic_predicated_cfg_i32_status[3] = guard_failures;
  generic_predicated_cfg_i32_status[4] = nested_checksum;
  generic_predicated_cfg_i32_status[5] = reversed_checksum;
  generic_predicated_cfg_i32_status[6] = direct_checksum;
  generic_predicated_cfg_i32_status[7] =
      static_cast<uint32_t>(generic_predicated_cfg_i32_nested[kGuardElements]);
  generic_predicated_cfg_i32_status[8] = static_cast<uint32_t>(
      generic_predicated_cfg_i32_reversed[kGuardElements + 3]);
  generic_predicated_cfg_i32_status[9] = static_cast<uint32_t>(
      generic_predicated_cfg_i32_direct[kGuardElements]);
  generic_predicated_cfg_i32_status[10] = 3;
  generic_predicated_cfg_i32_status[11] = 3;
  generic_predicated_cfg_i32_status[12] = direct_active;
  generic_predicated_cfg_i32_status[13] = 0;
  generic_predicated_cfg_i32_status[14] = 0;
  generic_predicated_cfg_i32_status[15] = UINT32_C(0x47434647); // GCFG

  linxi_puts("=== generic_predicated_cfg_i32 ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
