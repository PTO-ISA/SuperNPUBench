#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) || \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "selected_radix_tile_element requires the public ElementTile API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace selected_radix_tile_element_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElementCount = 263;
constexpr std::size_t kBlockElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kPaddedCount = 384;
constexpr std::size_t kBinCount = 256;
constexpr uint32_t kSelectedHighDigit = 0xa5u;

// The Tile operations derive radix digits, the element-wise region performs
// a normal conditional atomic, and a final Tile operation transforms the
// atomic return values before storage.
__attribute__((noinline)) void
selected_radix_tile_element(const uint32_t *input, std::size_t count,
                            uint32_t selected_high, uint32_t *histogram,
                            uint32_t *old_plus_seven) {
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid = count - begin < kBlockElements
                                  ? count - begin
                                  : kBlockElements;

    ElementTile<uint32_t, kBlockElements> input_tile;
    TLOAD(input_tile, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(input_tile, valid);

    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> values;
      ElementTile<uint32_t, kPartElements> high_digits;
      ElementTile<uint32_t, kPartElements> low_digits;
      ElementTile<uint32_t, kPartElements> atomic_old;
      ElementTile<uint32_t, kPartElements> transformed_old;

      auto input_elements = parts.part(part);
      TADDS(values, input_elements, 0u);
      TSHRS(high_digits, values, 8u);
      TANDS(low_digits, values, 0xffu);

      // 下标表示当前分区内的逻辑元素；物理布局由正式 API 和编译器管理。
      auto &predicate_elements = TPARTELEMENT(high_digits);
      auto &bucket_elements = TPARTELEMENT(low_digits);
      auto &old_elements = TPARTELEMENT(atomic_old);
      const uint32_t selected = selected_high;
      const uint32_t valid_elements =
          static_cast<uint32_t>(parts.valid_size(part));

// 此循环按元素生成谓词和 Tile 指令；未选中的元素不触发原子访存。
#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements &&
            predicate_elements[element] == selected) {
          old_elements[element] = __atomic_fetch_add(
              &histogram[bucket_elements[element]], 1u, __ATOMIC_RELAXED);
        } else {
          old_elements[element] = 0u;
        }
      }

      // 与上面的 element-wise 区域交替：这里继续对整块 Tile 做运算。
      TADDS(transformed_old, atomic_old, 7u);
      TSTORE(old_plus_seven + begin, transformed_old, parts, part);
    }
  }
}

} // namespace selected_radix_tile_element_benchmark

using namespace selected_radix_tile_element_benchmark;

extern "C" {
alignas(4096) uint32_t selected_radix_tile_element_input[kPaddedCount];
alignas(4096) uint32_t selected_radix_tile_element_histogram[kBinCount];
alignas(4096) uint32_t selected_radix_tile_element_old_plus_seven[kPaddedCount];
alignas(32) uint32_t selected_radix_tile_element_status[8];
}

int main() {
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    uint32_t value = UINT32_MAX;
    if (element < kElementCount) {
      uint32_t high = (element * 29u + 3u) & 0xffu;
      if ((element % 7u) < 3u)
        high = kSelectedHighDigit;
      if (high == kSelectedHighDigit && (element % 7u) >= 3u)
        high = 0x5au;
      uint32_t low = (element * 73u + (element / 9u) * 11u) & 0xffu;
      if ((element % 5u) < 3u)
        low = 0x2au;
      if (element == 0u)
        low = 0u;
      if (element == 1u)
        low = 255u;
      value = (high << 8u) | low;
    }
    selected_radix_tile_element_input[element] = value;
    selected_radix_tile_element_old_plus_seven[element] = 0xdeadbeefu;
  }
  for (std::size_t bin = 0; bin < kBinCount; ++bin)
    selected_radix_tile_element_histogram[bin] = 0u;

  BENCHSTART;
  selected_radix_tile_element(selected_radix_tile_element_input,
                              kElementCount, kSelectedHighDigit,
                              selected_radix_tile_element_histogram,
                              selected_radix_tile_element_old_plus_seven);
  BENCHEND;

  uint32_t expected[kBinCount] = {};
  uint32_t selected_count = 0u;
  for (std::size_t element = 0; element < kElementCount; ++element) {
    const uint32_t value = selected_radix_tile_element_input[element];
    if ((value >> 8u) == kSelectedHighDigit) {
      ++expected[value & 0xffu];
      ++selected_count;
    }
  }

  uint32_t failures = 0u;
  uint32_t total = 0u;
  uint32_t checksum = 0u;
  for (std::size_t bin = 0; bin < kBinCount; ++bin) {
    const uint32_t observed = selected_radix_tile_element_histogram[bin];
    failures += observed != expected[bin];
    total += observed;
    checksum += static_cast<uint32_t>(bin) * observed;
  }

  selected_radix_tile_element_status[0] = selected_count;
  selected_radix_tile_element_status[1] = failures;
  selected_radix_tile_element_status[2] = total;
  selected_radix_tile_element_status[3] =
      selected_radix_tile_element_histogram[0];
  selected_radix_tile_element_status[4] =
      selected_radix_tile_element_histogram[255];
  selected_radix_tile_element_status[5] = checksum;
  selected_radix_tile_element_status[6] = kSelectedHighDigit;
  selected_radix_tile_element_status[7] = 7u;

  linxi_puts("=== selected_radix_tile_element ===");
  linxi_put_kv("selected_elements", selected_count);
  linxi_put_kv("failures", failures);
  linxi_puts(failures == 0u ? "PASS" : "FAIL");
  return failures == 0u ? 0 : 1;
}
