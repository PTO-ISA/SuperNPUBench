#include "single_thread/sort/element_atomic_topk.hpp"

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace topk_boundaries_benchmark {

using element_atomic_topk_kernel::TopKResult;
using element_atomic_topk_kernel::TopKWorkspace;
using element_atomic_topk_kernel::topk16;

constexpr std::size_t kRadix = 256;
constexpr std::size_t kMaxCount = 129;
constexpr std::size_t kOldCount = 256;
constexpr std::size_t kOutputCount = kMaxCount + 4;
constexpr uint32_t kSentinel = 0xdeadbeefu;
constexpr uint32_t kMixed = 0u;
constexpr uint32_t kEqual = 1u;
constexpr uint32_t kTies = 2u;

struct BoundaryCase {
  uint32_t count;
  uint32_t requested_k;
  uint32_t pattern;
};

constexpr std::size_t kCaseCount = 17;
#ifndef TOPK_BOUNDARY_FIRST
#define TOPK_BOUNDARY_FIRST 0
#endif
#ifndef TOPK_BOUNDARY_LAST
#define TOPK_BOUNDARY_LAST kCaseCount
#endif
#ifndef TOPK_BOUNDARY_SHARED_WORKSPACE
#define TOPK_BOUNDARY_SHARED_WORKSPACE 0
#endif
constexpr std::size_t kFirstCase = TOPK_BOUNDARY_FIRST;
constexpr std::size_t kLastCase = TOPK_BOUNDARY_LAST;
static_assert(kFirstCase < kLastCase && kLastCase <= kCaseCount);

} // namespace topk_boundaries_benchmark

using namespace topk_boundaries_benchmark;

extern "C" {
alignas(32) BoundaryCase topk_boundaries_cases[kCaseCount] = {
    {0, 0, kMixed},     {0, 7, kMixed},     {129, 0, kMixed},
    {129, 1, kMixed},   {129, 129, kMixed}, {129, 222, kMixed},
    {1, 1, kMixed},     {31, 17, kMixed},   {32, 17, kMixed},
    {33, 17, kMixed},   {127, 37, kMixed},  {128, 37, kMixed},
    {129, 37, kMixed},  {10, 3, kTies},     {129, 1, kEqual},
    {129, 65, kEqual},  {129, 129, kEqual},
};
alignas(4096) uint32_t topk_boundaries_input[kMaxCount];
alignas(4096) uint32_t topk_boundaries_high_hist[kRadix];
alignas(4096) uint32_t topk_boundaries_low_hist[kRadix];
alignas(4096) uint32_t topk_boundaries_old_high[kOldCount];
alignas(4096) uint32_t topk_boundaries_old_low[kOldCount];
alignas(4096) uint32_t topk_boundaries_output[kOutputCount];
alignas(32) uint32_t topk_boundaries_status[kCaseCount];
}

namespace {

alignas(4096) uint8_t old_seen[kRadix][kMaxCount] = {};

struct CaseBuffers {
  alignas(4096) uint32_t input[kMaxCount];
  alignas(4096) uint32_t high_hist[kRadix];
  alignas(4096) uint32_t low_hist[kRadix];
  alignas(4096) uint32_t old_high[kOldCount];
  alignas(4096) uint32_t old_low[kOldCount];
  alignas(4096) uint32_t output[kOutputCount];
};

alignas(4096) CaseBuffers case_buffers[kCaseCount];

uint32_t tie_value(std::size_t element) {
  constexpr uint32_t values[10] = {
      0xffffu, 0xf100u, 0xf100u, 0xf100u, 0xf100u,
      0xe999u, 0xe999u, 0x8000u, 0x0100u, 0x0000u,
  };
  return values[element];
}

void prepare_input(uint32_t *input, const BoundaryCase &test_case) {
  for (std::size_t element = 0; element < kMaxCount; ++element) {
    uint32_t value = kSentinel;
    if (element < test_case.count) {
      if (test_case.pattern == kEqual) {
        value = 0x9a7cu;
      } else if (test_case.pattern == kTies) {
        value = tie_value(element);
      } else {
        value = static_cast<uint32_t>(
            (element * 40503u + (element / 17u) * 257u +
             (element % 11u) * 4099u) &
            0xffffu);
        if (element == 0u)
          value = 0xffffu;
        if (element == 1u)
          value = 0u;
      }
    }
    input[element] = value;
  }
}

void sort_descending(uint32_t *values, std::size_t count) {
  for (std::size_t current = 1; current < count; ++current) {
    const uint32_t value = values[current];
    std::size_t position = current;
    while (position != 0 && values[position - 1] < value) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
}

uint32_t verify_old_values(const uint32_t *input, const uint32_t *old_values,
                           const uint32_t *histogram,
                           std::size_t count, unsigned shift,
                           bool selected_only, uint32_t selected_high,
                           uint8_t generation) {
  uint32_t failures = 0u;
  for (std::size_t element = 0; element < count; ++element) {
    const uint32_t value = input[element];
    const bool active =
        !selected_only || ((value >> 8u) & 0xffu) == selected_high;
    const uint32_t old = old_values[element];
    if (!active) {
      failures += old != 0u;
      continue;
    }
    const uint32_t bin = (value >> shift) & 0xffu;
    failures += old >= histogram[bin];
    if (old < kMaxCount) {
      failures += old_seen[bin][old] == generation;
      old_seen[bin][old] = generation;
    }
  }

  const std::size_t padded = ((count + 127u) / 128u) * 128u;
  for (std::size_t element = count; element < padded; ++element)
    failures += old_values[element] != 0u;
  for (std::size_t element = padded; element < kOldCount; ++element)
    failures += old_values[element] != kSentinel;
  return failures;
}

uint32_t verify_case(const BoundaryCase &test_case, const TopKResult &result,
                     CaseBuffers &buffers, uint8_t generation) {
  const std::size_t count = test_case.count;
  const std::size_t actual_k =
      test_case.requested_k < count ? test_case.requested_k : count;
  uint32_t expected_high[kRadix] = {};
  uint32_t expected_low[kRadix] = {};
  uint32_t expected[kMaxCount] = {};
  for (std::size_t element = 0; element < count; ++element) {
    const uint32_t value = buffers.input[element];
    ++expected_high[(value >> 8u) & 0xffu];
    expected[element] = value;
  }
  sort_descending(expected, count);

  uint32_t failures = 0u;
  failures += result.output_count != actual_k;
  failures += actual_k == 0u ? result.cutoff != UINT16_MAX
                             : result.cutoff != expected[actual_k - 1u];
  failures += actual_k == 0u ? result.selected_high != 255u
                             : result.selected_high !=
                                   (expected[actual_k - 1u] >> 8u);

  std::size_t strictly_greater = 0u;
  if (actual_k != 0u) {
    for (std::size_t element = 0; element < count; ++element)
      strictly_greater +=
          buffers.input[element] > expected[actual_k - 1u];
  }
  failures += result.strictly_greater != strictly_greater;
  failures += result.equal_needed != actual_k - strictly_greater;

  if (actual_k != 0u) {
    for (std::size_t element = 0; element < count; ++element) {
      const uint32_t value = buffers.input[element];
      if (((value >> 8u) & 0xffu) == result.selected_high)
        ++expected_low[value & 0xffu];
    }
  }
  for (std::size_t bin = 0; bin < kRadix; ++bin) {
    failures += buffers.high_hist[bin] != expected_high[bin];
    failures += buffers.low_hist[bin] != expected_low[bin];
  }

  sort_descending(buffers.output, actual_k);
  for (std::size_t element = 0; element < actual_k; ++element)
    failures += buffers.output[element] != expected[element];
  for (std::size_t element = actual_k; element < kOutputCount; ++element)
    failures += buffers.output[element] != kSentinel;

  failures += verify_old_values(buffers.input, buffers.old_high,
                                buffers.high_hist, count, 8u,
                                false, 0u,
                                static_cast<uint8_t>(generation * 2u + 1u));
  failures += verify_old_values(buffers.input, buffers.old_low,
                                buffers.low_hist, count, 0u,
                                true,
                                actual_k == 0u ? 256u
                                               : result.selected_high,
                                static_cast<uint8_t>(generation * 2u + 2u));
  return failures;
}

} // namespace

int main() {
  uint32_t total_failures = 0u;
  BENCHSTART;
#pragma clang loop unroll(disable)
  for (std::size_t case_index = kFirstCase; case_index < kLastCase;
       ++case_index) {
    const BoundaryCase test_case = topk_boundaries_cases[case_index];
#if TOPK_BOUNDARY_SHARED_WORKSPACE
    CaseBuffers &buffers = case_buffers[0];
#else
    CaseBuffers &buffers = case_buffers[case_index];
#endif
    prepare_input(buffers.input, test_case);
    for (std::size_t bin = 0; bin < kRadix; ++bin) {
      buffers.high_hist[bin] = kSentinel;
      buffers.low_hist[bin] = kSentinel;
    }
    for (std::size_t element = 0; element < kOldCount; ++element) {
      buffers.old_high[element] = kSentinel;
      buffers.old_low[element] = kSentinel;
    }
    for (std::size_t element = 0; element < kOutputCount; ++element)
      buffers.output[element] = kSentinel;

    TopKWorkspace workspace{
        buffers.high_hist, buffers.low_hist,
        buffers.old_high, buffers.old_low,
    };
    const TopKResult result =
        topk16(buffers.input, test_case.count,
               test_case.requested_k, buffers.output, workspace);
    const uint32_t failures = verify_case(
        test_case, result, buffers, static_cast<uint8_t>(case_index));
    topk_boundaries_status[case_index] = failures;
    total_failures += failures;
  }
  BENCHEND;

#if TOPK_BOUNDARY_SHARED_WORKSPACE
  const CaseBuffers &final_buffers = case_buffers[0];
#else
  const CaseBuffers &final_buffers = case_buffers[kLastCase - 1u];
#endif
  for (std::size_t element = 0; element < kMaxCount; ++element)
    topk_boundaries_input[element] = final_buffers.input[element];
  for (std::size_t bin = 0; bin < kRadix; ++bin) {
    topk_boundaries_high_hist[bin] = final_buffers.high_hist[bin];
    topk_boundaries_low_hist[bin] = final_buffers.low_hist[bin];
  }
  for (std::size_t element = 0; element < kOldCount; ++element) {
    topk_boundaries_old_high[element] = final_buffers.old_high[element];
    topk_boundaries_old_low[element] = final_buffers.old_low[element];
  }
  for (std::size_t element = 0; element < kOutputCount; ++element)
    topk_boundaries_output[element] = final_buffers.output[element];

  linxi_puts("=== topk_boundaries ===");
  linxi_put_kv("cases", kLastCase - kFirstCase);
  linxi_put_kv("failures", total_failures);
  linxi_puts(total_failures == 0u ? "PASS" : "FAIL");
  return total_failures == 0u ? 0 : 1;
}
