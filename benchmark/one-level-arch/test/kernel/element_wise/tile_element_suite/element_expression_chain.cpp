#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_ELEMENT_TILE) || \
    PTO_TILEOP_API_HAS_ELEMENT_TILE != 1
#error "element_expression_chain requires the public ElementTile API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace element_expression_chain_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElementCount = 263;
constexpr std::size_t kBlockElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kPaddedCount = 384;

// 一个完整 kernel 交替使用 TileOp 和普通 C++ 元素表达式。参数和类型只描述
// 逻辑元素；Tile 的物理布局、表达式临时值和中间 Tile 均由 API 与编译器管理。
__attribute__((noinline)) void element_expression_chain(
    const uint32_t *input, std::size_t count, uint32_t bias,
    uint32_t multiplier, uint32_t subtract, uint32_t divisor,
    uint32_t modulus, uint32_t left_shift, uint32_t right_shift,
    uint32_t mask, uint32_t or_value, uint32_t xor_value,
    uint32_t *arithmetic_output, uint32_t *bitwise_output,
    uint32_t *unary_output) {
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid = count - begin < kBlockElements
                                  ? count - begin
                                  : kBlockElements;

    ElementTile<uint32_t, kBlockElements> input_tile;
    TLOAD(input_tile, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(input_tile, valid);

    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<uint32_t, kPartElements> values;
      ElementTile<uint32_t, kPartElements> arithmetic;
      ElementTile<uint32_t, kPartElements> bitwise;
      ElementTile<uint32_t, kPartElements> unary;

      auto input_elements = parts.part(part);
      TADDS(values, input_elements, 0u);

      // element array 的下标是当前分区内的逻辑元素编号。
      // 循环里的局部变量由编译器分析并分配 intermediate Tile。
      auto &value_elements = TPARTELEMENT(values);
      auto &arithmetic_elements = TPARTELEMENT(arithmetic);

#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        uint32_t biased = value_elements[element] + bias;
        uint32_t multiplied = biased * multiplier;
        uint32_t reduced = multiplied - subtract;
        uint32_t divided = reduced / divisor;
        uint32_t remainder = divided % modulus;
        arithmetic_elements[element] = remainder;
      }
      TSTORE(arithmetic_output + begin, arithmetic, parts, part);

      auto &bitwise_elements = TPARTELEMENT(bitwise);

#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        uint32_t shifted_left = value_elements[element] << left_shift;
        uint32_t shifted_right = shifted_left >> right_shift;
        uint32_t masked = shifted_right & mask;
        uint32_t merged = masked | or_value;
        uint32_t mixed = merged ^ xor_value;
        // 复用早期结果，要求 shifted_left 跨多个 intermediate Tile 保持存活。
        uint32_t reused = mixed ^ shifted_left;
        bitwise_elements[element] = reused;
      }
      TSTORE(bitwise_output + begin, bitwise, parts, part);

      auto &unary_elements = TPARTELEMENT(unary);

#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        uint32_t negated = -value_elements[element];
        uint32_t inverted = ~negated;
        unary_elements[element] = inverted;
      }
      TSTORE(unary_output + begin, unary, parts, part);
    }
  }
}

} // namespace element_expression_chain_benchmark

using namespace element_expression_chain_benchmark;

extern "C" {
alignas(4096) uint32_t element_expression_chain_input[kPaddedCount];
alignas(4096) uint32_t element_expression_chain_arithmetic[kPaddedCount];
alignas(4096) uint32_t element_expression_chain_bitwise[kPaddedCount];
alignas(4096) uint32_t element_expression_chain_unary[kPaddedCount];
alignas(32) uint32_t element_expression_chain_status[8];
}

int main() {
  constexpr uint32_t kBias = 19u;
  constexpr uint32_t kMultiplier = 13u;
  constexpr uint32_t kSubtract = 7u;
  constexpr uint32_t kDivisor = 5u;
  constexpr uint32_t kModulus = 251u;
  constexpr uint32_t kLeftShift = 3u;
  constexpr uint32_t kRightShift = 2u;
  constexpr uint32_t kMask = 0xfffffff0u;
  constexpr uint32_t kOrValue = 0x00000105u;
  constexpr uint32_t kXorValue = 0x0055aa11u;

  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    uint32_t value = 0xdeadbeefu;
    if (element < kElementCount) {
      value = static_cast<uint32_t>(element) * 2654435761u + 0x01234567u;
      value ^= static_cast<uint32_t>(element / 7u) * 0x00010101u;
    }
    element_expression_chain_input[element] = value;
    element_expression_chain_arithmetic[element] = 0xdeadbeefu;
    element_expression_chain_bitwise[element] = 0xdeadbeefu;
    element_expression_chain_unary[element] = 0xdeadbeefu;
  }

  BENCHSTART;
  element_expression_chain(
      element_expression_chain_input, kElementCount, kBias, kMultiplier,
      kSubtract, kDivisor, kModulus, kLeftShift, kRightShift, kMask,
      kOrValue, kXorValue, element_expression_chain_arithmetic,
      element_expression_chain_bitwise, element_expression_chain_unary);
  BENCHEND;

  uint32_t failures = 0u;
  uint32_t arithmetic_checksum = 0u;
  uint32_t bitwise_checksum = 0u;
  uint32_t unary_checksum = 0u;
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    uint32_t value = element < kElementCount
                         ? element_expression_chain_input[element]
                         : 0u;
    uint32_t expected_arithmetic =
        (((value + kBias) * kMultiplier - kSubtract) / kDivisor) %
        kModulus;
    uint32_t expected_bitwise =
        ((((value << kLeftShift) >> kRightShift) & kMask) | kOrValue) ^
        kXorValue ^ (value << kLeftShift);
    uint32_t expected_unary = ~static_cast<uint32_t>(-value);
    failures +=
        element_expression_chain_arithmetic[element] != expected_arithmetic;
    failures += element_expression_chain_bitwise[element] != expected_bitwise;
    failures += element_expression_chain_unary[element] != expected_unary;
    arithmetic_checksum += element_expression_chain_arithmetic[element];
    bitwise_checksum += element_expression_chain_bitwise[element];
    unary_checksum += element_expression_chain_unary[element];
  }

  element_expression_chain_status[0] = static_cast<uint32_t>(kElementCount);
  element_expression_chain_status[1] = failures;
  element_expression_chain_status[2] = arithmetic_checksum;
  element_expression_chain_status[3] = bitwise_checksum;
  element_expression_chain_status[4] = unary_checksum;
  element_expression_chain_status[5] = kPaddedCount;
  element_expression_chain_status[6] = 10u;
  element_expression_chain_status[7] = 0x45585052u; // "EXPR"

  linxi_puts("=== element_expression_chain ===");
  linxi_put_kv("elements", kElementCount);
  linxi_put_kv("failures", failures);
  linxi_puts(failures == 0u ? "PASS" : "FAIL");
  return failures == 0u ? 0 : 1;
}
