#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) || \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "indexed_gather_tile_element requires the public ElementTile API"
#endif

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_VIEW_METADATA) || \
    PTO_TILEOP_API_HAS_ELEMENT_VIEW_METADATA != 1
#error "indexed gather requires the public logical element-view metadata"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace indexed_gather_tile_element_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElementCount = 263;
constexpr std::size_t kBlockElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kPaddedCount = 384;
constexpr std::size_t kTableElements = 257;
constexpr std::size_t kGuardElements = 16;
constexpr std::size_t kOutputElements =
    kGuardElements + kPaddedCount + kGuardElements;
constexpr uint32_t kGuardValue = 0x6a09e667u;

// 一个完整 kernel 交替使用 TileOp 和普通 C++ 下标表达式。程序只声明逻辑
// ElementTile；物理布局、字节地址换算、执行掩码和中间 Tile 都由 API 与编译器管理。
__attribute__((noinline)) void indexed_gather_tile_element(
    const uint32_t *__restrict table, const uint32_t *__restrict indices,
    std::size_t count, uint32_t *__restrict output) {
  for (std::size_t begin = 0; begin < kPaddedCount;
       begin += kBlockElements) {
    const std::size_t valid =
        begin < count
            ? (count - begin < kBlockElements ? count - begin : kBlockElements)
            : 0u;

    ElementTile<uint32_t, kBlockElements> index_tile;
    // 故意加载完整 padded index。尾部保持 UINT32_MAX poison，用来证明元素 if
    // 产生的执行掩码在访存前生效，而不是靠 TLOAD 把无效 index 清零。
    TLOAD(index_tile, indices + begin, kBlockElements);
    auto parts = TPARTVIEW<kPartElements>(index_tile, valid);

    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> index_copy;
      ElementTile<uint32_t, kPartElements> gathered;
      ElementTile<uint32_t, kPartElements> postprocessed;

      auto index_part = parts.part(part);
      TADDS(index_copy, index_part, 0u);

      auto &index_elements = TPARTELEMENT(index_copy);
      auto &output_elements = TPARTELEMENT(gathered);
      const uint32_t valid_elements =
          static_cast<uint32_t>(parts.valid_size(part));

// 普通数组表达式描述按元素 gather；无效元素不读取 poison index。
#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        if (element < valid_elements)
          output_elements[element] = table[index_elements[element]];
        else
          output_elements[element] = 0u;
      }

      // gather 的 Tile 结果继续进入正式 TileOp，并写满 padded 输出区域。
      TADDS(postprocessed, gathered, 3u);
      TSTORE(output + begin, postprocessed, parts, part);
    }
  }
}

} // namespace indexed_gather_tile_element_benchmark

using namespace indexed_gather_tile_element_benchmark;

extern "C" {
alignas(4096) uint32_t indexed_gather_tile_element_table[kTableElements];
alignas(4096) uint32_t indexed_gather_tile_element_indices[kPaddedCount];
alignas(4096) uint32_t indexed_gather_tile_element_output[kOutputElements];
alignas(32) uint32_t indexed_gather_tile_element_status[12];
}

int main() {
  for (std::size_t element = 0; element < kTableElements; ++element) {
    const uint32_t mixed =
        static_cast<uint32_t>(element) * 0x01020305u + 0x10203040u;
    indexed_gather_tile_element_table[element] =
        0x80000000u | (mixed & 0x7fffffffu);
  }

  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    uint32_t index = UINT32_MAX;
    if (element < kElementCount) {
      index = static_cast<uint32_t>(
          (element * 73u + (element / 5u) * 19u) % kTableElements);
      if (element % 11u < 4u)
        index = 42u;
      if (element == 0u)
        index = 0u;
      if (element == kElementCount - 1u)
        index = static_cast<uint32_t>(kTableElements - 1u);
    }
    indexed_gather_tile_element_indices[element] = index;
  }
  for (std::size_t element = 0; element < kOutputElements; ++element)
    indexed_gather_tile_element_output[element] = kGuardValue;

  // 先运行 257 个有效元素：最后一个 block 的后三个完整 part 都无效，且它们的
  // 96 个 index 全是 UINT32_MAX。该调用专门验证全空 GPR mask 不会发起访存。
  constexpr std::size_t kEmptyPartProbeCount = 257;
  indexed_gather_tile_element(
      indexed_gather_tile_element_table,
      indexed_gather_tile_element_indices, kEmptyPartProbeCount,
      indexed_gather_tile_element_output + kGuardElements);
  uint32_t probe_failures = 0u;
  uint32_t probe_guard_failures = 0u;
  uint32_t probe_empty_part_failures = 0u;
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    const uint32_t expected =
        element < kEmptyPartProbeCount
            ? indexed_gather_tile_element_table[
                  indexed_gather_tile_element_indices[element]] +
                  3u
            : 3u;
    const uint32_t observed =
        indexed_gather_tile_element_output[kGuardElements + element];
    probe_failures += observed != expected;
    if (element >= 288u)
      probe_empty_part_failures += observed != 3u;
  }
  for (std::size_t element = 0; element < kGuardElements; ++element) {
    probe_guard_failures +=
        indexed_gather_tile_element_output[element] != kGuardValue;
    probe_guard_failures +=
        indexed_gather_tile_element_output[kGuardElements + kPaddedCount +
                                           element] != kGuardValue;
  }
  probe_failures += probe_guard_failures;

  for (std::size_t element = 0; element < kOutputElements; ++element)
    indexed_gather_tile_element_output[element] = kGuardValue;

  BENCHSTART;
  indexed_gather_tile_element(
      indexed_gather_tile_element_table,
      indexed_gather_tile_element_indices, kElementCount,
      indexed_gather_tile_element_output + kGuardElements);
  BENCHEND;

  uint32_t failures = 0u;
  uint32_t active_checksum = 0u;
  uint32_t padded_checksum = 0u;
  uint32_t guard_failures = 0u;
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    const uint32_t expected =
        element < kElementCount
            ? indexed_gather_tile_element_table[
                  indexed_gather_tile_element_indices[element]] +
                  3u
            : 3u;
    const uint32_t observed =
        indexed_gather_tile_element_output[kGuardElements + element];
    failures += observed != expected;
    if (element < kElementCount)
      active_checksum += observed;
    padded_checksum += observed;
  }
  for (std::size_t element = 0; element < kGuardElements; ++element) {
    guard_failures +=
        indexed_gather_tile_element_output[element] != kGuardValue;
    guard_failures +=
        indexed_gather_tile_element_output[kGuardElements + kPaddedCount +
                                           element] != kGuardValue;
  }
  failures += guard_failures;
  failures += probe_failures;

  indexed_gather_tile_element_status[0] =
      static_cast<uint32_t>(kElementCount);
  indexed_gather_tile_element_status[1] = failures;
  indexed_gather_tile_element_status[2] = active_checksum;
  indexed_gather_tile_element_status[3] = padded_checksum;
  indexed_gather_tile_element_status[4] = guard_failures;
  indexed_gather_tile_element_status[5] =
      static_cast<uint32_t>(kTableElements);
  indexed_gather_tile_element_status[6] =
      static_cast<uint32_t>(kPaddedCount);
  indexed_gather_tile_element_status[7] = 0x47415448u; // "GATH"
  indexed_gather_tile_element_status[8] =
      static_cast<uint32_t>(kEmptyPartProbeCount);
  indexed_gather_tile_element_status[9] = probe_failures;
  indexed_gather_tile_element_status[10] = probe_guard_failures;
  indexed_gather_tile_element_status[11] = probe_empty_part_failures;

  linxi_puts("=== indexed_gather_tile_element ===");
  linxi_put_kv("elements", kElementCount);
  linxi_put_kv("failures", failures);
  linxi_puts(failures == 0u ? "PASS" : "FAIL");
  return failures == 0u ? 0 : 1;
}
