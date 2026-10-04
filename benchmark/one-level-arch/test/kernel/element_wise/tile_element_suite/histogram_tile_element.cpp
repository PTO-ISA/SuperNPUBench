#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) || \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "histogram_tile_element requires the public ElementTile API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace histogram_tile_element_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElementCount = 263;
constexpr std::size_t kBlockElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kPaddedCount = 384;
constexpr std::size_t kBinCount = 256;

// A complete TILE -> element-wise -> TILE kernel. The source expresses only
// logical element counts; the adapter/compiler own the physical Tile layout.
__attribute__((noinline)) void
histogram_tile_element(const uint32_t *input, std::size_t count,
                       uint32_t *histogram, uint32_t *old_plus_one) {
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid = count - begin < kBlockElements
                                  ? count - begin
                                  : kBlockElements;

    ElementTile<uint32_t, kBlockElements> input_tile;
    TLOAD(input_tile, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(input_tile, valid);

    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> values;
      ElementTile<uint32_t, kPartElements> buckets;
      ElementTile<uint32_t, kPartElements> atomic_old;
      ElementTile<uint32_t, kPartElements> transformed_old;

      auto input_elements = parts.part(part);
      TADDS(values, input_elements, 0u);
      TANDS(buckets, values, 0xffu);

      auto &bucket_elements = TPARTELEMENT(buckets);
      auto &old_elements = TPARTELEMENT(atomic_old);
      const uint32_t valid_elements =
          static_cast<uint32_t>(parts.valid_size(part));

#pragma linx elementwise
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements) {
          old_elements[element] = __atomic_fetch_add(
              &histogram[bucket_elements[element]], 1u, __ATOMIC_RELAXED);
        } else {
          old_elements[element] = 0u;
        }
      }

      TADDS(transformed_old, atomic_old, 1u);
      TSTORE(old_plus_one + begin, transformed_old, parts, part);
    }
  }
}

} // namespace histogram_tile_element_benchmark

using namespace histogram_tile_element_benchmark;

extern "C" {
alignas(4096) uint32_t histogram_tile_element_input[kPaddedCount];
alignas(4096) uint32_t histogram_tile_element_histogram[kBinCount];
alignas(4096) uint32_t histogram_tile_element_old_plus_one[kPaddedCount];
alignas(32) uint32_t histogram_tile_element_status[8];
}

int main() {
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    uint32_t value = UINT32_MAX;
    if (element < kElementCount) {
      value = (element * 73u + (element / 9u) * 11u) & 0xffu;
      if ((element & 3u) < 2u)
        value = 0x2au;
      if (element == 0u || element == 128u)
        value = 0u;
      if (element == 1u || element == 262u)
        value = 255u;
    }
    histogram_tile_element_input[element] = value;
    histogram_tile_element_old_plus_one[element] = 0xdeadbeefu;
  }
  for (std::size_t bin = 0; bin < kBinCount; ++bin)
    histogram_tile_element_histogram[bin] = 0u;

  BENCHSTART;
  histogram_tile_element(histogram_tile_element_input, kElementCount,
                         histogram_tile_element_histogram,
                         histogram_tile_element_old_plus_one);
  BENCHEND;

  uint32_t expected[kBinCount] = {};
  for (std::size_t element = 0; element < kElementCount; ++element)
    ++expected[histogram_tile_element_input[element] & 0xffu];

  uint32_t failures = 0u;
  uint32_t total = 0u;
  uint32_t checksum = 0u;
  for (std::size_t bin = 0; bin < kBinCount; ++bin) {
    const uint32_t observed = histogram_tile_element_histogram[bin];
    failures += observed != expected[bin];
    total += observed;
    checksum += static_cast<uint32_t>(bin) * observed;
  }

  histogram_tile_element_status[0] = static_cast<uint32_t>(kElementCount);
  histogram_tile_element_status[1] = failures;
  histogram_tile_element_status[2] = total;
  histogram_tile_element_status[3] = histogram_tile_element_histogram[0];
  histogram_tile_element_status[4] = histogram_tile_element_histogram[255];
  histogram_tile_element_status[5] = checksum;
  histogram_tile_element_status[6] = 1u;
  histogram_tile_element_status[7] = 0x48495354u; // "HIST"

  linxi_puts("=== histogram_tile_element ===");
  linxi_put_kv("elements", kElementCount);
  linxi_put_kv("failures", failures);
  linxi_puts(failures == 0u ? "PASS" : "FAIL");
  return failures == 0u ? 0 : 1;
}
