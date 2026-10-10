#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) ||                               \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "elementwise_atomic_histogram requires the public ElementTile API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace elementwise_atomic_histogram_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kBins = 8;

// 公共 element API 使用 U32 承载谓词。benchmark 对外保留原始 byte 谓词数组，
// 在进入 kernel 前将其拓宽到 U32 transport 数组。
alignas(4096) static uint32_t active_transport[kElements];

// 先完成全部原子更新，再进入只读 gather 阶段。即使同一个标记循环内各 element
// 的执行顺序由实现决定，也能保持原程序先 scatter-add、后 gather 的语义。
__attribute__((noinline)) void elementwise_atomic_histogram(
    const int32_t *__restrict values, const int32_t *__restrict indices,
    const uint32_t *__restrict active, int32_t *__restrict bins,
    uint32_t *__restrict observed) {
  ElementTile<int32_t, kElements> value_tile;
  ElementTile<int32_t, kElements> index_tile;
  ElementTile<uint32_t, kElements> active_tile;

  TLOAD(value_tile, values, kElements);
  TLOAD(index_tile, indices, kElements);
  TLOAD(active_tile, active, kElements);

  auto value_parts = TPARTVIEW<kPartElements>(value_tile, kElements);
  auto index_parts = TPARTVIEW<kPartElements>(index_tile, kElements);
  auto active_parts = TPARTVIEW<kPartElements>(active_tile, kElements);

  // 阶段一：四个 32-element part 全部完成 histogram 原子更新。
  for (std::size_t part = 0; part < value_parts.size(); ++part) {
    ElementTile<int32_t, kPartElements> part_values;
    ElementTile<int32_t, kPartElements> part_indices;
    ElementTile<int32_t, kPartElements> atomic_old;

    auto value_part = value_parts.part(part);
    auto index_part = index_parts.part(part);
    TADDS(part_values, value_part, 0);
    TADDS(part_indices, index_part, 0);

    auto &value_elements = TPARTELEMENT(part_values);
    auto &index_elements = TPARTELEMENT(part_indices);
    auto &old_elements = TPARTELEMENT(atomic_old);

#pragma pto element for
    for (unsigned element = 0; element < kPartElements; ++element) {
      old_elements[element] = __atomic_fetch_add(
          bins + index_elements[element], value_elements[element],
          __ATOMIC_RELAXED);
    }
  }

  // 阶段二：读取已经更新完成的 bin。禁用 element 返回原始的零 padding seed，
  // 并且不对 bins 发起读取。
  constexpr uint32_t kDisabledSeed = 0u;
  for (std::size_t part = 0; part < index_parts.size(); ++part) {
    ElementTile<int32_t, kPartElements> part_indices;
    ElementTile<uint32_t, kPartElements> part_active;
    ElementTile<uint32_t, kPartElements> gathered;

    auto index_part = index_parts.part(part);
    auto active_part = active_parts.part(part);
    TADDS(part_indices, index_part, 0);
    TADDS(part_active, active_part, 0u);

    auto &index_elements = TPARTELEMENT(part_indices);
    auto &active_elements = TPARTELEMENT(part_active);
    auto &gathered_elements = TPARTELEMENT(gathered);

#pragma pto element for
    for (unsigned element = 0; element < kPartElements; ++element) {
      if (active_elements[element] != 0u) {
        gathered_elements[element] =
            static_cast<uint32_t>(bins[index_elements[element]]);
      } else {
        gathered_elements[element] = kDisabledSeed;
      }
    }

    TSTORE(observed, gathered, active_parts, part);
  }
}

} // namespace elementwise_atomic_histogram_benchmark

using namespace elementwise_atomic_histogram_benchmark;

extern "C" {
alignas(4096) int32_t elementwise_atomic_histogram_bins[kBins];
alignas(4096) int32_t elementwise_atomic_histogram_values[kElements];
alignas(4096) int32_t elementwise_atomic_histogram_indices[kElements];
alignas(4096) unsigned char elementwise_atomic_histogram_active[kElements];
alignas(4096) uint32_t elementwise_atomic_histogram_observed[kElements];
alignas(32) uint32_t elementwise_atomic_histogram_status[8];
}

int main() {
  for (std::size_t bin = 0; bin < kBins; ++bin)
    elementwise_atomic_histogram_bins[bin] = 0;

  for (std::size_t element = 0; element < kElements; ++element) {
    elementwise_atomic_histogram_values[element] = 1;
    elementwise_atomic_histogram_indices[element] =
        static_cast<int32_t>(element & 7u);
    elementwise_atomic_histogram_active[element] = 1u;
    active_transport[element] =
        static_cast<uint32_t>(elementwise_atomic_histogram_active[element]);
    elementwise_atomic_histogram_observed[element] = 0u;
  }

  BENCHSTART;
  elementwise_atomic_histogram(
      elementwise_atomic_histogram_values,
      elementwise_atomic_histogram_indices, active_transport,
      elementwise_atomic_histogram_bins,
      elementwise_atomic_histogram_observed);
  BENCHEND;

  uint32_t failures = 0u;
  uint32_t bin_total = 0u;
  uint32_t observed_total = 0u;
  uint32_t active_count = 0u;
  for (std::size_t bin = 0; bin < kBins; ++bin) {
    const uint32_t result =
        static_cast<uint32_t>(elementwise_atomic_histogram_bins[bin]);
    failures += result != 16u;
    bin_total += result;
  }
  for (std::size_t element = 0; element < kElements; ++element) {
    failures += elementwise_atomic_histogram_values[element] != 1;
    failures += elementwise_atomic_histogram_indices[element] !=
                static_cast<int32_t>(element & 7u);
    failures += elementwise_atomic_histogram_active[element] != 1u;
    failures += elementwise_atomic_histogram_observed[element] != 16u;
    observed_total += elementwise_atomic_histogram_observed[element];
    active_count += elementwise_atomic_histogram_active[element] != 0u;
  }

  elementwise_atomic_histogram_status[0] = kElements;
  elementwise_atomic_histogram_status[1] = kBins;
  elementwise_atomic_histogram_status[2] = failures;
  elementwise_atomic_histogram_status[3] = bin_total;
  elementwise_atomic_histogram_status[4] = observed_total;
  elementwise_atomic_histogram_status[5] = active_count;
  elementwise_atomic_histogram_status[6] = kElements / kPartElements;
  elementwise_atomic_histogram_status[7] = 0x41544853u; // "ATHS"

  linxi_puts("=== elementwise_atomic_histogram ===");
  linxi_put_kv("failures", failures);
  return failures == 0u ? 0 : 1;
}
