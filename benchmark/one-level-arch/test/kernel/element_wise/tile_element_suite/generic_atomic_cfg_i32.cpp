#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) ||                               \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "generic_atomic_cfg_i32 requires the public ElementTile API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace generic_atomic_cfg_i32_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kValidElements = 125;
constexpr std::size_t kBins = 32;
constexpr std::size_t kCenter = 16;
constexpr uint32_t kPoisonSeed = 0x6c8e9cf5u;

// 一个完整 kernel：TileOp 准备有符号下标和增量，普通 C++ CFG 执行原子
// 更新，随后 TileOp 继续处理两个 atomic old value。不同 element 可以无序，
// 但同一个 element 的两次 atomic_fetch_add 必须保持源码顺序。
__attribute__((noinline)) void generic_atomic_cfg_i32(
    const int32_t *__restrict indices, const int32_t *__restrict deltas,
    const uint32_t *__restrict inactive_gates,
    uint32_t *__restrict histogram_center,
    uint32_t *__restrict inactive_null_base, uint32_t *__restrict first_output,
    uint32_t *__restrict second_output, uint32_t *__restrict combined_output,
    uint32_t *__restrict poison_output) {
  ElementTile<int32_t, kElements> loaded_indices;
  ElementTile<int32_t, kElements> loaded_deltas;
  ElementTile<uint32_t, kElements> loaded_gates;

  TLOAD(loaded_indices, indices, kElements);
  TLOAD(loaded_deltas, deltas, kElements);
  TLOAD(loaded_gates, inactive_gates, kElements);
  auto index_parts = TPARTVIEW<kPartElements>(loaded_indices, kValidElements);
  auto delta_parts = TPARTVIEW<kPartElements>(loaded_deltas, kValidElements);
  auto gate_parts = TPARTVIEW<kPartElements>(loaded_gates, kValidElements);

  for (std::size_t part = 0; part < index_parts.size(); ++part) {
    ElementTile<int32_t, kPartElements> prepared_indices;
    ElementTile<int32_t, kPartElements> prepared_deltas;
    ElementTile<uint32_t, kPartElements> prepared_gates;
    ElementTile<uint32_t, kPartElements> first_old;
    ElementTile<uint32_t, kPartElements> second_old;
    ElementTile<uint32_t, kPartElements> combined;
    ElementTile<uint32_t, kPartElements> poison_old;

    auto index_part = index_parts.part(part);
    auto delta_part = delta_parts.part(part);
    auto gate_part = gate_parts.part(part);
    TADDS(prepared_indices, index_part, 0);
    TADDS(prepared_deltas, delta_part, 0);
    TADDS(prepared_gates, gate_part, 0u);
    TADDS(first_old, prepared_gates, 0u);
    TADDS(second_old, prepared_gates, 0u);
    TADDS(combined, prepared_gates, 0u);
    TADDS(poison_old, prepared_gates, kPoisonSeed);

    auto &index_elements = TPARTELEMENT(prepared_indices);
    auto &delta_elements = TPARTELEMENT(prepared_deltas);
    auto &gate_elements = TPARTELEMENT(prepared_gates);
    auto &first_elements = TPARTELEMENT(first_old);
    auto &second_elements = TPARTELEMENT(second_old);
    auto &combined_elements = TPARTELEMENT(combined);
    auto &poison_elements = TPARTELEMENT(poison_old);
    const uint32_t valid_elements =
        static_cast<uint32_t>(index_parts.valid_size(part));

// 普通基本块和 PHI 选择每个 element 的非单位增量。负增量按 uint32_t
// 模 2^32 参与原子加；有符号负下标由 TLEA 先符号扩展、再换算成字节偏移。
#pragma pto element for
    for (unsigned element = 0; element < kPartElements; ++element) {
      if (element < valid_elements) {
        const int32_t index = index_elements[element];
        const int32_t delta = delta_elements[element];
        uint32_t first_add;
        uint32_t second_add;
        if (index < 0) {
          if (delta < 0) {
            first_add = static_cast<uint32_t>(delta);
            second_add = 5u;
          } else {
            first_add = static_cast<uint32_t>(delta + 2);
            second_add = static_cast<uint32_t>(-3);
          }
        } else if ((index & 1) == 0) {
          first_add = static_cast<uint32_t>(delta - 4);
          second_add = 9u;
        } else {
          first_add = static_cast<uint32_t>(delta + 6);
          second_add = static_cast<uint32_t>(-7);
        }

        const uint32_t old_first = __atomic_fetch_add(
            histogram_center + index, first_add, __ATOMIC_RELAXED);
        const uint32_t old_second = __atomic_fetch_add(
            histogram_center + index, second_add, __ATOMIC_RELAXED);
        first_elements[element] = old_first;
        second_elements[element] = old_second;
        combined_elements[element] = old_first ^ (old_second + 0x13579bdfu);
      }

      // gates 全为零；这里故意组合 null base 和 poison 下标，验证全空执行
      // mask 不会向 TLSU 发出原子访存，也不会覆盖预先准备的 Tile 值。
      if (gate_elements[element] != 0u) {
        poison_elements[element] =
            __atomic_fetch_add(inactive_null_base + index_elements[element],
                               13u, __ATOMIC_RELAXED);
      }
    }

    ElementTile<uint32_t, kPartElements> first_post;
    ElementTile<uint32_t, kPartElements> second_post;
    ElementTile<uint32_t, kPartElements> combined_post;
    ElementTile<uint32_t, kPartElements> poison_post;
    TADDS(first_post, first_old, 11u);
    TADDS(second_post, second_old, 13u);
    TADDS(combined_post, combined, 17u);
    TADDS(poison_post, poison_old, 1u);
    TSTORE(first_output, first_post, gate_parts, part);
    TSTORE(second_output, second_post, gate_parts, part);
    TSTORE(combined_output, combined_post, gate_parts, part);
    TSTORE(poison_output, poison_post, gate_parts, part);
  }
}

} // namespace generic_atomic_cfg_i32_benchmark

using namespace generic_atomic_cfg_i32_benchmark;

extern "C" {
alignas(4096) int32_t generic_atomic_cfg_i32_indices[kElements];
alignas(4096) int32_t generic_atomic_cfg_i32_deltas[kElements];
alignas(4096) uint32_t generic_atomic_cfg_i32_gates[kElements];
alignas(4096) uint32_t generic_atomic_cfg_i32_histogram[kBins];
alignas(4096) uint32_t generic_atomic_cfg_i32_first[kElements];
alignas(4096) uint32_t generic_atomic_cfg_i32_second[kElements];
alignas(4096) uint32_t generic_atomic_cfg_i32_combined[kElements];
alignas(4096) uint32_t generic_atomic_cfg_i32_poison[kElements];
alignas(32) uint32_t generic_atomic_cfg_i32_status[8];
}

int main() {
  for (std::size_t element = 0; element < kElements; ++element) {
    generic_atomic_cfg_i32_indices[element] =
        element < kValidElements
            ? static_cast<int32_t>((element * 5u + 3u) % kBins) -
                  static_cast<int32_t>(kCenter)
            : INT32_MIN;
    generic_atomic_cfg_i32_deltas[element] =
        element < kValidElements
            ? static_cast<int32_t>((element * 7u + 2u) % 17u) - 8
            : INT32_MIN;
    generic_atomic_cfg_i32_gates[element] = 0u;
    generic_atomic_cfg_i32_first[element] = 0xdeadbeefu;
    generic_atomic_cfg_i32_second[element] = 0xdeadbeefu;
    generic_atomic_cfg_i32_combined[element] = 0xdeadbeefu;
    generic_atomic_cfg_i32_poison[element] = 0xdeadbeefu;
  }
  for (std::size_t bin = 0; bin < kBins; ++bin)
    generic_atomic_cfg_i32_histogram[bin] =
        0x10203040u + static_cast<uint32_t>(bin) * 0x101u;

  BENCHSTART;
  generic_atomic_cfg_i32(
      generic_atomic_cfg_i32_indices, generic_atomic_cfg_i32_deltas,
      generic_atomic_cfg_i32_gates, generic_atomic_cfg_i32_histogram + kCenter,
      nullptr, generic_atomic_cfg_i32_first, generic_atomic_cfg_i32_second,
      generic_atomic_cfg_i32_combined, generic_atomic_cfg_i32_poison);
  BENCHEND;

  uint32_t expected[kBins];
  for (std::size_t bin = 0; bin < kBins; ++bin)
    expected[bin] = 0x10203040u + static_cast<uint32_t>(bin) * 0x101u;
  for (std::size_t element = 0; element < kValidElements; ++element) {
    const int32_t index = generic_atomic_cfg_i32_indices[element];
    const int32_t delta = generic_atomic_cfg_i32_deltas[element];
    uint32_t first_add;
    uint32_t second_add;
    if (index < 0) {
      if (delta < 0) {
        first_add = static_cast<uint32_t>(delta);
        second_add = 5u;
      } else {
        first_add = static_cast<uint32_t>(delta + 2);
        second_add = static_cast<uint32_t>(-3);
      }
    } else if ((index & 1) == 0) {
      first_add = static_cast<uint32_t>(delta - 4);
      second_add = 9u;
    } else {
      first_add = static_cast<uint32_t>(delta + 6);
      second_add = static_cast<uint32_t>(-7);
    }
    expected[static_cast<std::size_t>(index + kCenter)] +=
        first_add + second_add;
  }

  uint32_t failures = 0u;
  uint32_t histogram_checksum = 0u;
  uint32_t poison_checksum = 0u;
  for (std::size_t bin = 0; bin < kBins; ++bin) {
    failures += generic_atomic_cfg_i32_histogram[bin] != expected[bin];
    histogram_checksum += generic_atomic_cfg_i32_histogram[bin];
  }
  for (std::size_t element = 0; element < kElements; ++element) {
    const uint32_t expected_poison = kPoisonSeed + 1u;
    failures += generic_atomic_cfg_i32_poison[element] != expected_poison;
    poison_checksum += generic_atomic_cfg_i32_poison[element];
  }

  generic_atomic_cfg_i32_status[0] = kElements;
  generic_atomic_cfg_i32_status[1] = kValidElements;
  generic_atomic_cfg_i32_status[2] = failures;
  generic_atomic_cfg_i32_status[3] = histogram_checksum;
  generic_atomic_cfg_i32_status[4] = poison_checksum;
  generic_atomic_cfg_i32_status[5] = kBins;
  generic_atomic_cfg_i32_status[6] = 2u;
  generic_atomic_cfg_i32_status[7] = 0x41544d43u; // "ATMC"

  linxi_puts("=== generic_atomic_cfg_i32 ===");
  linxi_put_kv("failures", failures);
  return failures == 0u ? 0 : 1;
}
