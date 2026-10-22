#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                        \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "generic_typed_tile_cfg_i32 requires the official typed element API"
#endif

#include "benchmark.h"
#include "linx_print.h"

#include <cstddef>
#include <cstdint>

namespace generic_typed_tile_cfg_i32_benchmark {

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElementCount = 263;
constexpr std::size_t kBlockElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr std::size_t kPaddedCount = 384;
constexpr std::size_t kGuardElements = 16;
constexpr int32_t kGuardValue = INT32_C(0x5a5a6b6b);

// 完整 kernel 只使用正式 TileOp API 和普通 C++ 控制流。程序员看到的是
// 逻辑元素；物理 layout、lane、谓词 Tile 和中间 Tile 都由 API 与编译器管理。
__attribute__((noinline)) void
generic_typed_tile_cfg_i32(const int32_t *__restrict input, std::size_t count,
                           int32_t *__restrict output) {
  for (std::size_t begin = 0; begin < count; begin += kBlockElements) {
    const std::size_t valid =
        count - begin < kBlockElements ? count - begin : kBlockElements;

    ElementTile<int32_t, kBlockElements> loaded;
    TLOAD(loaded, input + begin, valid);
    auto parts = TPARTVIEW<kPartElements>(loaded, valid);

    for (std::size_t part = 0; part < parts.size(); ++part) {
      ElementTile<int32_t, kPartElements> prepared;
      ElementTile<int32_t, kPartElements> selected;
      ElementTile<int32_t, kPartElements> biased;
      ElementTile<int32_t, kPartElements> result;

      // TileOp 先准备数据，再把当前逻辑分区交给 element for。
      auto input_part = parts.part(part);
      TADDS(prepared, input_part, int32_t(3));
      auto &prepared_elements = TPARTELEMENT(prepared);
      auto &selected_elements = TPARTELEMENT(selected);

// 这是标准 C++ nested if。每个元素内部保持源码顺序；不同元素可无序执行。
#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        const int32_t value = prepared_elements[element];
        if (value < 0) {
          selected_elements[element] = value / 3;
        } else if (value == 0) {
          selected_elements[element] = 7;
        } else {
          selected_elements[element] = value ^ 5;
        }
      }

      // element for 的结果继续流入 TileOp；TMOV 复制完整的值和 definedness。
      TADDS(biased, selected, int32_t(11));
      TMOV(result, biased);
      auto &result_elements = TPARTELEMENT(result);

// 第二个区域只覆盖部分元素。没有进入任一分支的元素必须保留 TMOV 写入的
// 非零 seed，不能被未初始化值、零值或前一个 element region 覆盖。
#pragma pto element for
      for (unsigned element = 0; element < kPartElements; ++element) {
        const int32_t current = result_elements[element];
        if (current < -100) {
          result_elements[element] = current ^ INT32_C(0x55);
        } else if (current > 100) {
          result_elements[element] = current - 4;
        }
      }

      TSTORE(output + begin, result, parts, part);
    }
  }
}

} // namespace generic_typed_tile_cfg_i32_benchmark

using namespace generic_typed_tile_cfg_i32_benchmark;

extern "C" {
alignas(4096) int32_t generic_typed_tile_cfg_i32_input[kPaddedCount];
alignas(4096) int32_t
    generic_typed_tile_cfg_i32_output[kPaddedCount + 2 * kGuardElements];
alignas(32) uint32_t generic_typed_tile_cfg_i32_status[12];
}

static int32_t expected_value(int32_t input) {
  const int32_t prepared = input + 3;
  int32_t selected;
  if (prepared < 0)
    selected = prepared / 3;
  else if (prepared == 0)
    selected = 7;
  else
    selected = prepared ^ 5;

  int32_t result = selected + 11;
  if (result < -100)
    result ^= INT32_C(0x55);
  else if (result > 100)
    result -= 4;
  return result;
}

int main() {
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    int32_t value = INT32_C(0x1234567);
    if (element < kElementCount)
      value =
          static_cast<int32_t>((element * 7919u + 12345u) % 200001u) - 100000;
    if (element == 0)
      value = -3;
    if (element == 1)
      value = -4;
    if (element == 2)
      value = 0;
    if (element == 3)
      value = 100;
    generic_typed_tile_cfg_i32_input[element] = value;
  }
  for (std::size_t element = 0; element < kPaddedCount + 2 * kGuardElements;
       ++element)
    generic_typed_tile_cfg_i32_output[element] = kGuardValue;

  BENCHSTART;
  generic_typed_tile_cfg_i32(generic_typed_tile_cfg_i32_input, kElementCount,
                             generic_typed_tile_cfg_i32_output +
                                 kGuardElements);
  BENCHEND;

  uint32_t failures = 0;
  uint32_t guard_failures = 0;
  uint32_t checksum = 0;
  uint32_t retained = 0;
  for (std::size_t element = 0; element < kPaddedCount; ++element) {
    const int32_t input =
        element < kElementCount ? generic_typed_tile_cfg_i32_input[element] : 0;
    const int32_t expected = expected_value(input);
    const int32_t observed =
        generic_typed_tile_cfg_i32_output[kGuardElements + element];
    failures += observed != expected;
    checksum += static_cast<uint32_t>(observed);
    const int32_t prepared = input + 3;
    const int32_t selected =
        prepared < 0 ? prepared / 3 : (prepared == 0 ? 7 : (prepared ^ 5));
    retained += selected + 11 >= -100 && selected + 11 <= 100;
  }
  for (std::size_t element = 0; element < kGuardElements; ++element) {
    guard_failures += generic_typed_tile_cfg_i32_output[element] != kGuardValue;
    guard_failures +=
        generic_typed_tile_cfg_i32_output[kGuardElements + kPaddedCount +
                                          element] != kGuardValue;
  }
  failures += guard_failures;

  generic_typed_tile_cfg_i32_status[0] = kElementCount;
  generic_typed_tile_cfg_i32_status[1] = kPaddedCount;
  generic_typed_tile_cfg_i32_status[2] = failures;
  generic_typed_tile_cfg_i32_status[3] = guard_failures;
  generic_typed_tile_cfg_i32_status[4] = checksum;
  generic_typed_tile_cfg_i32_status[5] =
      static_cast<uint32_t>(generic_typed_tile_cfg_i32_output[kGuardElements]);
  generic_typed_tile_cfg_i32_status[6] = static_cast<uint32_t>(
      generic_typed_tile_cfg_i32_output[kGuardElements + 1]);
  generic_typed_tile_cfg_i32_status[7] = static_cast<uint32_t>(
      generic_typed_tile_cfg_i32_output[kGuardElements + 2]);
  generic_typed_tile_cfg_i32_status[8] = retained;
  generic_typed_tile_cfg_i32_status[9] = 2;
  generic_typed_tile_cfg_i32_status[10] = 0;
  generic_typed_tile_cfg_i32_status[11] = UINT32_C(0x54594346); // TYCF

  linxi_puts("=== generic_typed_tile_cfg_i32 ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
