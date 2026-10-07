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

// 输入使用 U32 保存 0..65535 的键；两张 histogram 各含 256 个计数器。
// 原子旧值缓冲按 128 个元素补齐；输入无需额外 padding，输出是无序 Top-K 集合。
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

  // 第一轮：统计所有元素的高 8 位。
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
      // element array 借用 Tile 的逻辑元素，不复制主存数组或暴露物理布局。
      auto &bucket_elements = TPARTELEMENT(buckets);
      auto &old_elements = TPARTELEMENT(old);
      const uint32_t valid_elements = static_cast<uint32_t>(parts.valid_size(part));
// 只对有效、被选中的逻辑元素生成原子效果；由编译器生成谓词与字节索引。
#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements) {
          old_elements[element] = __atomic_fetch_add(
              &high_hist[bucket_elements[element]], 1u, __ATOMIC_RELAXED);
        } else {
          old_elements[element] = 0;
        }
      }
      // 写回位置对应原输入的逻辑元素顺序。
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

  // 第二轮：只对选中的高位 bucket 统计低 8 位。
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
      // element array 借用 Tile 的逻辑元素，不复制主存数组或暴露物理布局。
      auto &bucket_elements = TPARTELEMENT(buckets);
      auto &predicate_elements = TPARTELEMENT(predicate);
      auto &old_elements = TPARTELEMENT(old);
      const uint32_t valid_elements = static_cast<uint32_t>(parts.valid_size(part));
// 只对有效、被选中的逻辑元素生成原子效果；由编译器生成谓词与字节索引。
#pragma pto element for
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
