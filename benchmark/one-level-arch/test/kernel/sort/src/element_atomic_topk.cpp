#include "single_thread/sort/element_atomic_topk.hpp"

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

using namespace element_atomic_topk_kernel;

constexpr std::size_t kCount = 777;
constexpr std::size_t kPaddedCount =
    ((kCount + kParentLanes - 1) / kParentLanes) * kParentLanes;
constexpr std::size_t kRequestedK = 37;
constexpr std::size_t kRadix = 256;

extern "C" {
alignas(4096) uint32_t element_atomic_topk_input[kPaddedCount];
alignas(4096) uint32_t element_atomic_topk_high_hist[kRadix];
alignas(4096) uint32_t element_atomic_topk_low_hist[kRadix];
alignas(4096) uint32_t element_atomic_topk_old_high[kPaddedCount];
alignas(4096) uint32_t element_atomic_topk_old_low[kPaddedCount];
alignas(4096) uint32_t element_atomic_topk_output[kRequestedK];
alignas(32) uint32_t element_atomic_topk_status[8];
}

namespace {

uint32_t (&input)[kPaddedCount] = element_atomic_topk_input;
uint32_t (&high_hist)[kRadix] = element_atomic_topk_high_hist;
uint32_t (&low_hist)[kRadix] = element_atomic_topk_low_hist;
uint32_t (&old_high)[kPaddedCount] = element_atomic_topk_old_high;
uint32_t (&old_low)[kPaddedCount] = element_atomic_topk_old_low;
uint32_t (&output)[kRequestedK] = element_atomic_topk_output;
alignas(4096) static uint32_t expected_high_hist[kRadix];
alignas(4096) static uint32_t expected_low_hist[kRadix];
alignas(4096) static uint32_t expected[kRequestedK];
alignas(4096) static uint32_t high_bins[kPaddedCount];
alignas(4096) static uint32_t low_bins[kPaddedCount];
constexpr std::size_t kOldSeenBytes = (kCount + 7U) / 8U;
alignas(4096) static uint8_t old_seen[kRadix][kOldSeenBytes];

void make_input() {
  for (std::size_t i = 0; i < kPaddedCount; ++i) {
    uint32_t value = UINT32_MAX;
    if (i < kCount) {
      value = static_cast<uint32_t>(
          (i * 40503u + (i / 17u) * 257u) & 0xffffu);
      if ((i >= 250U && i < 270U) || (i >= 510U && i < 526U)) {
        value = 0xf123u;
      }
      if (i == 0U || i == 256U || i == 512U) {
        value = 0xffffu;
      }
    }
    input[i] = value;
  }
}

void clear(uint32_t *values, std::size_t count, uint32_t fill = 0) {
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = fill;
  }
}

uint8_t descending_bucket(const uint32_t *hist, std::size_t k,
                          std::size_t *greater) {
  std::size_t accumulated = 0;
  for (int bucket = 255; bucket >= 0; --bucket) {
    if (accumulated + hist[bucket] >= k) {
      *greater = accumulated;
      return static_cast<uint8_t>(bucket);
    }
    accumulated += hist[bucket];
  }
  *greater = accumulated;
  return 0;
}

void independent_golden() {
  clear(expected_high_hist, kRadix);
  for (std::size_t i = 0; i < kCount; ++i) {
    ++expected_high_hist[(input[i] >> 8) & 0xffu];
  }

  // Direct insertion top-k is deliberately independent from the radix
  // implementation under test.
  clear(expected, kRequestedK);
  std::size_t used = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    const uint32_t value = input[i];
    std::size_t position = used;
    while (position != 0 && expected[position - 1] < value) {
      if (position < kRequestedK) {
        expected[position] = expected[position - 1];
      }
      --position;
    }
    if (position < kRequestedK) {
      expected[position] = value;
      if (used < kRequestedK) {
        ++used;
      }
    }
  }
}

int verify_atomic_old_values(const uint32_t *old_values, const uint32_t *bins,
                             const uint32_t *histogram, bool selected_only,
                             uint32_t selected_high) {
  const auto old_storage_index = [](std::size_t logical_index) {
    const std::size_t parent = logical_index / kParentLanes;
    const std::size_t in_parent = logical_index % kParentLanes;
    const std::size_t part = in_parent % kParts;
    const std::size_t lane = in_parent / kParts;
    return parent * kParentLanes + part * kLanes + lane;
  };
  int failures = 0;
  for (std::size_t bin = 0; bin < kRadix; ++bin) {
    for (std::size_t byte = 0; byte < kOldSeenBytes; ++byte) {
      old_seen[bin][byte] = 0;
    }
  }
  for (std::size_t i = 0; i < kCount; ++i) {
    const bool selected =
        !selected_only || (((input[i] >> 8) & 0xffu) == selected_high);
    const uint32_t old = old_values[old_storage_index(i)];
    if (!selected) {
      failures += old != 0;
      continue;
    }
    const uint32_t bin = bins[i];
    failures += old >= histogram[bin];
    if (old < histogram[bin]) {
      const uint8_t bit = static_cast<uint8_t>(1U << (old & 7U));
      failures += (old_seen[bin][old >> 3U] & bit) != 0U;
      old_seen[bin][old >> 3U] |= bit;
    }
  }
  for (std::size_t bin = 0; bin < kRadix; ++bin) {
    // An atomic fetch-add over a bin containing n updates must return
    // every old value in 0..n-1 exactly once, independent of lane order.
    for (uint32_t expected_old = 0; expected_old < histogram[bin];
         ++expected_old) {
      const uint8_t bit = static_cast<uint8_t>(1U << (expected_old & 7U));
      failures += (old_seen[bin][expected_old >> 3U] & bit) == 0U;
    }
  }
  for (std::size_t i = kCount; i < kPaddedCount; ++i) {
    failures += old_values[old_storage_index(i)] != 0;
  }
  return failures;
}

} // namespace

int main() {
  make_input();
  independent_golden();
  clear(high_hist, kRadix);
  clear(low_hist, kRadix);
  clear(old_high, kPaddedCount);
  clear(old_low, kPaddedCount);

  BENCHSTART;
  histogram_high8(input, kCount, high_hist, old_high);

  std::size_t greater_high = 0;
  const uint8_t selected_high =
      descending_bucket(high_hist, kRequestedK, &greater_high);
  histogram_selected_low8(input, kCount, selected_high, low_hist, old_low);

  std::size_t greater_low = 0;
  const uint8_t selected_low =
      descending_bucket(low_hist, kRequestedK - greater_high, &greater_low);
  const uint16_t cutoff = static_cast<uint16_t>(
      (static_cast<uint16_t>(selected_high) << 8) | selected_low);
  const std::size_t strictly_greater = greater_high + greater_low;
  std::size_t equal_remaining = kRequestedK - strictly_greater;
  std::size_t output_count = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    if (input[i] > cutoff) {
      if (output_count < kRequestedK) {
        output[output_count] = input[i];
      }
      ++output_count;
    }
  }
  for (std::size_t i = 0; i < kCount && equal_remaining != 0; ++i) {
    if (input[i] == cutoff) {
      if (output_count < kRequestedK) {
        output[output_count] = input[i];
      }
      ++output_count;
      --equal_remaining;
    }
  }
  BENCHEND;

  clear(expected_low_hist, kRadix);
  for (std::size_t i = 0; i < kCount; ++i) {
    high_bins[i] = (input[i] >> 8) & 0xffu;
    low_bins[i] = input[i] & 0xffu;
    if (high_bins[i] == selected_high) {
      ++expected_low_hist[low_bins[i]];
    }
  }

  int failures = 0;
  uint32_t failure_mask = 0U;
  const int shape_failures = output_count != kRequestedK || equal_remaining != 0;
  failures += shape_failures;
  failure_mask |= shape_failures != 0 ? 0x1U : 0U;
  int histogram_failures = 0;
  uint32_t observed_high_total = 0U;
  uint32_t observed_low_total = 0U;
  for (std::size_t bin = 0; bin < kRadix; ++bin) {
    observed_high_total += high_hist[bin];
    observed_low_total += low_hist[bin];
    histogram_failures += high_hist[bin] != expected_high_hist[bin];
    histogram_failures += low_hist[bin] != expected_low_hist[bin];
  }
  failures += histogram_failures;
  failure_mask |= histogram_failures != 0 ? 0x2U : 0U;
  const int old_high_failures = verify_atomic_old_values(
      old_high, high_bins, high_hist, false, selected_high);
  const int old_low_failures = verify_atomic_old_values(
      old_low, low_bins, low_hist, true, selected_high);
  failures += old_high_failures + old_low_failures;
  failure_mask |= old_high_failures != 0 ? 0x4U : 0U;
  failure_mask |= old_low_failures != 0 ? 0x8U : 0U;

  // Compare multisets, including a cutoff tie, without depending on which
  // equal-key positions were selected.
  const std::size_t stored_output_count =
      output_count < kRequestedK ? output_count : kRequestedK;
  for (std::size_t i = 0; i < stored_output_count; ++i) {
    for (std::size_t j = i + 1; j < stored_output_count; ++j) {
      if (output[i] < output[j]) {
        const uint32_t temporary = output[i];
        output[i] = output[j];
        output[j] = temporary;
      }
    }
  }
  int output_failures = 0;
  for (std::size_t i = 0; i < kRequestedK; ++i) {
    output_failures += output[i] != expected[i];
  }
  failures += output_failures;
  failure_mask |= output_failures != 0 ? 0x10U : 0U;

  linxi_puts("=== element_atomic_topk ===");
  linxi_put_kv("count", kCount);
  linxi_put_kv("output_count", output_count);
  linxi_put_kv("failures", failures);
  linxi_puts(failures == 0 ? "PASS" : "FAIL");
  element_atomic_topk_status[0] = static_cast<uint32_t>(output_count);
  element_atomic_topk_status[1] = failure_mask;
  element_atomic_topk_status[2] = cutoff;
  element_atomic_topk_status[3] = selected_high;
  element_atomic_topk_status[4] = observed_high_total;
  element_atomic_topk_status[5] = observed_low_total;
  element_atomic_topk_status[6] = high_hist[255];
  element_atomic_topk_status[7] = high_hist[0];
  const uint32_t diagnostic_return = failure_mask |
      ((static_cast<uint32_t>(output_count) & 0x3ffU) << 6U) |
      ((observed_high_total & 0x3ffU) << 16U);
  return failure_mask == 0U ? 0 : static_cast<int>(diagnostic_return);
}
