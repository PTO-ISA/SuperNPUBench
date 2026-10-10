#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                       \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "concat_gather_s32_original requires the official typed element API"
#endif

#if !defined(PTO_TILEOP_API_HAS_512_INTEGER_ELEMENT_TILE) ||                 \
    PTO_TILEOP_API_HAS_512_INTEGER_ELEMENT_TILE != 1
#error "concat_gather_s32_original requires the official 512-element integer tile API"
#endif

#include "benchmark.h"
#include "linx_print.h"
#include "single_thread/concat/concat_gather.hpp"

#include <cstddef>
#include <cstdint>

#ifndef CONCAT_GATHER_S32_ALIAS_PROBE
#define CONCAT_GATHER_S32_ALIAS_PROBE 0
#endif

namespace concat_gather_s32_original_benchmark {

constexpr std::size_t kMaxDimensions = 8;
constexpr std::size_t kDataDimensions = 2;
constexpr std::size_t kConcatDimension = 1;
constexpr int kInputElements = 128000;
constexpr bool kAliasProbe = CONCAT_GATHER_S32_ALIAS_PROBE == 1;
constexpr int kOutputElements = kAliasProbe ? 512 : 128000;
constexpr int kTileElements = 512;
constexpr std::size_t kGuardElements = 16;
constexpr std::size_t kAliasOutputOffset = 2049;
constexpr int32_t kGuardValue = INT32_C(0x5a5a6b6b);

} // namespace concat_gather_s32_original_benchmark

using namespace concat_gather_s32_original_benchmark;

extern "C" {
alignas(4096) int32_t concat_gather_s32_original_input[kInputElements];
alignas(4096) int32_t concat_gather_s32_original_output
    [kGuardElements + kOutputElements + kGuardElements];
alignas(32) std::size_t
    concat_gather_s32_original_in_shape[kMaxDimensions];
alignas(32) std::size_t
    concat_gather_s32_original_out_shape[kMaxDimensions];
alignas(32) uint32_t concat_gather_s32_original_status[12];
}

int main() {
  for (std::size_t element = 0; element < kInputElements; ++element) {
    concat_gather_s32_original_input[element] = static_cast<int32_t>(
        (static_cast<uint32_t>(element) * 104729u + 12345u) & 0x7fffffffu);
  }
  for (std::size_t element = 0;
       element < kGuardElements + kOutputElements + kGuardElements;
       ++element) {
    concat_gather_s32_original_output[element] = kGuardValue;
  }
  for (std::size_t dimension = 0; dimension < kMaxDimensions; ++dimension) {
    concat_gather_s32_original_in_shape[dimension] = 0;
    concat_gather_s32_original_out_shape[dimension] = 0;
  }
  concat_gather_s32_original_in_shape[0] = 64;
  concat_gather_s32_original_in_shape[1] = 2;
  concat_gather_s32_original_out_shape[0] = 64;
  concat_gather_s32_original_out_shape[1] = 2000;

  BENCHSTART;
  if constexpr (kAliasProbe) {
    // Keep only this call visible for IR/object checks; TileOp helpers inline.
    [[clang::noinline]]
    concat_gather<int32_t, kMaxDimensions, kInputElements, kOutputElements,
                  kTileElements, kDataDimensions, kConcatDimension>(
        concat_gather_s32_original_input,
        concat_gather_s32_original_input + kAliasOutputOffset,
        concat_gather_s32_original_in_shape,
        concat_gather_s32_original_out_shape);
  } else {
    [[clang::noinline]]
    concat_gather<int32_t, kMaxDimensions, kInputElements, kOutputElements,
                  kTileElements, kDataDimensions, kConcatDimension>(
        concat_gather_s32_original_input,
        concat_gather_s32_original_output + kGuardElements,
        concat_gather_s32_original_in_shape,
        concat_gather_s32_original_out_shape);
  }
  BENCHEND;

  uint32_t output_failures = 0;
  for (uint32_t global = 0; global < kOutputElements; ++global) {
    const uint32_t row = (global / 2000u) % 64u;
    const uint32_t col = global % 2000u;
    const uint32_t input_no = col / 2u;
    const uint32_t element_index =
        input_no * (64u * 2u) + row * 2u + (col % 2u);
    const int32_t expected = static_cast<int32_t>(
        (element_index * 104729u + 12345u) & 0x7fffffffu);
    if constexpr (kAliasProbe) {
      output_failures += concat_gather_s32_original_input
                             [kAliasOutputOffset + global] != expected;
    } else {
      output_failures += concat_gather_s32_original_output
                             [kGuardElements + global] != expected;
    }
  }

  uint32_t guard_failures = 0;
  if constexpr (kAliasProbe) {
    for (std::size_t element = 0;
         element < kGuardElements + kOutputElements + kGuardElements;
         ++element) {
      guard_failures +=
          concat_gather_s32_original_output[element] != kGuardValue;
    }
  } else {
    for (std::size_t element = 0; element < kGuardElements; ++element) {
      guard_failures +=
          concat_gather_s32_original_output[element] != kGuardValue;
      guard_failures += concat_gather_s32_original_output
                            [kGuardElements + kOutputElements + element] !=
                        kGuardValue;
    }
  }

  uint32_t input_failures = 0;
  for (std::size_t element = 0; element < kInputElements; ++element) {
    int32_t expected = static_cast<int32_t>(
        (static_cast<uint32_t>(element) * 104729u + 12345u) & 0x7fffffffu);
    if constexpr (kAliasProbe) {
      if (element >= kAliasOutputOffset &&
          element < kAliasOutputOffset + kOutputElements) {
        const uint32_t global =
            static_cast<uint32_t>(element - kAliasOutputOffset);
        const uint32_t row = (global / 2000u) % 64u;
        const uint32_t col = global % 2000u;
        const uint32_t input_no = col / 2u;
        const uint32_t element_index =
            input_no * (64u * 2u) + row * 2u + (col % 2u);
        expected = static_cast<int32_t>(
            (element_index * 104729u + 12345u) & 0x7fffffffu);
      }
    }
    input_failures += concat_gather_s32_original_input[element] != expected;
  }

  uint32_t shape_failures = 0;
  shape_failures += concat_gather_s32_original_in_shape[0] != 64;
  shape_failures += concat_gather_s32_original_in_shape[1] != 2;
  shape_failures += concat_gather_s32_original_out_shape[0] != 64;
  shape_failures += concat_gather_s32_original_out_shape[1] != 2000;
  for (std::size_t dimension = 2; dimension < kMaxDimensions; ++dimension) {
    shape_failures += concat_gather_s32_original_in_shape[dimension] != 0;
    shape_failures += concat_gather_s32_original_out_shape[dimension] != 0;
  }

  const uint32_t failures = output_failures + guard_failures + input_failures +
                            shape_failures;
  concat_gather_s32_original_status[0] = failures;
  concat_gather_s32_original_status[1] = output_failures;
  concat_gather_s32_original_status[2] = guard_failures;
  concat_gather_s32_original_status[3] = input_failures;
  concat_gather_s32_original_status[4] = shape_failures;
  concat_gather_s32_original_status[5] = kInputElements;
  concat_gather_s32_original_status[6] = kOutputElements;
  concat_gather_s32_original_status[7] = kTileElements;
  concat_gather_s32_original_status[8] = 64;
  concat_gather_s32_original_status[9] = 2;
  concat_gather_s32_original_status[10] = 2000;
  concat_gather_s32_original_status[11] = 0x43475332u; // CGS2

  linxi_puts("=== concat_gather_s32_original ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
