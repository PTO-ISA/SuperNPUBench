#include "reference.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

template <std::size_t N>
void check_case(const std::array<uint32_t, N> &input, std::size_t k) {
  std::array<uint32_t, N> actual{};
  std::array<uint32_t, N> expected = input;
  uint32_t high_hist[element_atomic_topk::kRadix];
  uint32_t low_hist[element_atomic_topk::kRadix];
  element_atomic_topk::Cutoff cutoff{};

  const std::size_t actual_count = element_atomic_topk::reference_topk(
      input.data(), input.size(), k, actual.data(), high_hist, low_hist,
      &cutoff);
  const std::size_t expected_count = std::min(k, input.size());
  std::sort(expected.begin(), expected.end(), std::greater<uint32_t>());
  std::sort(actual.begin(), actual.begin() + actual_count,
            std::greater<uint32_t>());

  assert(actual_count == expected_count);
  assert(std::equal(actual.begin(), actual.begin() + actual_count,
                    expected.begin()));

  uint32_t high_total = 0;
  uint32_t low_total = 0;
  for (std::size_t bin = 0; bin < element_atomic_topk::kRadix; ++bin) {
    high_total += high_hist[bin];
    low_total += low_hist[bin];
  }
  assert(high_total == input.size());
  if (expected_count == 0) {
    assert(low_total == 0);
  } else {
    assert(low_total == high_hist[cutoff.key >> 8]);
    assert(cutoff.strictly_greater + cutoff.equal_needed == expected_count);
  }
}

} // namespace

int main() {
  std::array<uint32_t, 777> mixed{};
  for (std::size_t i = 0; i < mixed.size(); ++i) {
    // Exercise both radix levels, duplicate hot bins, and all three 256-lane
    // boundaries. Values remain in the documented 16-bit key domain.
    mixed[i] = static_cast<uint32_t>((i * 40503u + (i / 17u) * 257u) & 0xffffu);
  }
  for (std::size_t i = 250; i < 270; ++i) {
    mixed[i] = 0xf123u;
  }
  for (std::size_t i = 510; i < 526; ++i) {
    mixed[i] = 0xf123u;
  }
  mixed[0] = 0xffffu;
  mixed[256] = 0xffffu;
  mixed[512] = 0xffffu;

  check_case(mixed, 0);
  check_case(mixed, 1);
  check_case(mixed, 37);
  check_case(mixed, mixed.size());

  std::array<uint32_t, 263> tied{};
  tied.fill(0x9a7cu);
  check_case(tied, 0);
  check_case(tied, 1);
  check_case(tied, 131);
  check_case(tied, tied.size());
  return 0;
}
