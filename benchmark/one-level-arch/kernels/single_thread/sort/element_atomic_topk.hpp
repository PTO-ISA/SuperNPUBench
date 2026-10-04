#ifndef ELEMENT_ATOMIC_TOPK_HPP
#define ELEMENT_ATOMIC_TOPK_HPP

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

#ifndef PTO_TILEOP_API_HAS_ELEMENT_TILE
#error "Install the public TileOp ElementTile API before building this kernel"
#endif

namespace element_atomic_topk_kernel {
using pto::ElementTile;
using pto::TPARTVIEW;
using pto::TPARTELEMENT;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kParts = 4;
constexpr std::size_t kBlockElements = 128;

struct TopKWorkspace {
  uint32_t *high_hist;
  uint32_t *low_hist;
  uint32_t *old_high;
  uint32_t *old_low;
};

struct TopKResult {
  std::size_t output_count;
  uint16_t cutoff;
  uint8_t selected_high;
  std::size_t strictly_greater;
  std::size_t equal_needed;
};

// U32 storage for keys in 0..65535. Histogram buffers hold 256 counters;
// diagnostic old-value buffers hold round_up(count, 128) elements.
// Input need not be padded. Output is an unordered Top-K multiset.
inline TopKResult topk16(const uint32_t *input, std::size_t count,
                        std::size_t requested_k, uint32_t *output,
                        TopKWorkspace workspace) {
  const std::size_t k = requested_k < count ? requested_k : count;
  const std::size_t padded_count = ((count + 127) / 128) * 128;
  for (std::size_t bin = 0; bin < 256; ++bin) {
    workspace.high_hist[bin] = 0;
    workspace.low_hist[bin] = 0;
  }
  for (std::size_t element = 0; element < padded_count; ++element) {
    workspace.old_high[element] = 0;
    workspace.old_low[element] = 0;
  }

  // Histogram all high digits.
  uint32_t *high_hist = workspace.high_hist;
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid = count - begin < kBlockElements
                                ? count - begin : kBlockElements;
    ElementTile<uint32_t, kBlockElements> keys;
    TLOAD(keys, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(keys, valid);
    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> values, buckets, old;
      auto view = parts.part(part);
      TADDS(values, view, 0u);
      TSHRS(buckets, values, 8u);
      auto &bucket_elements = TPARTELEMENT(buckets);
      auto &old_elements = TPARTELEMENT(old);
      const uint32_t valid_elements = static_cast<uint32_t>(parts.valid_size(part));
#pragma linx elementwise
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements) {
          old_elements[element] = __atomic_fetch_add(
              &high_hist[bucket_elements[element]], 1u, __ATOMIC_RELAXED);
        } else {
          old_elements[element] = 0;
        }
      }
      TSTORE(workspace.old_high + begin, old, parts, part);
    }
  }
  if (k == 0)
    return {0, UINT16_MAX, 255, 0, 0};

  std::size_t remaining = k;
  uint32_t selected_high = 255;
  for (int bin = 255; bin >= 0; --bin) {
    if (remaining <= high_hist[bin]) {
      selected_high = static_cast<uint32_t>(bin);
      break;
    }
    remaining -= high_hist[bin];
  }

  // Histogram low digits only for elements in the selected high bucket.
  uint32_t *low_hist = workspace.low_hist;
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid = count - begin < kBlockElements
                                ? count - begin : kBlockElements;
    ElementTile<uint32_t, kBlockElements> keys;
    TLOAD(keys, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(keys, valid);
    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> values, buckets, predicate, old;
      auto view = parts.part(part);
      TADDS(values, view, 0u);
      TSHRS(predicate, values, 8u);
      TANDS(buckets, values, 0xffu);
      auto &bucket_elements = TPARTELEMENT(buckets);
      auto &predicate_elements = TPARTELEMENT(predicate);
      auto &old_elements = TPARTELEMENT(old);
      const uint32_t valid_elements = static_cast<uint32_t>(parts.valid_size(part));
#pragma linx elementwise
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements && predicate_elements[element] == selected_high) {
          old_elements[element] = __atomic_fetch_add(
              &low_hist[bucket_elements[element]], 1u, __ATOMIC_RELAXED);
        } else {
          old_elements[element] = 0;
        }
      }
      TSTORE(workspace.old_low + begin, old, parts, part);
    }
  }

  uint32_t selected_low = 255;
  for (int bin = 255; bin >= 0; --bin) {
    if (remaining <= low_hist[bin]) {
      selected_low = static_cast<uint32_t>(bin);
      break;
    }
    remaining -= low_hist[bin];
  }
  const uint16_t cutoff = static_cast<uint16_t>(
      (selected_high << 8) | selected_low);
  const std::size_t equal_needed = remaining;
  std::size_t written = 0;
  for (std::size_t element = 0; element < count; ++element) {
    if (input[element] > cutoff)
      output[written++] = input[element];
  }
  for (std::size_t element = 0; element < count && remaining != 0; ++element) {
    if (input[element] == cutoff) {
      output[written++] = input[element];
      --remaining;
    }
  }
  return {written, cutoff, static_cast<uint8_t>(selected_high),
          k - equal_needed, equal_needed};
}

} // namespace element_atomic_topk_kernel
#endif
