#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                       \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "concat_gather_f16_original requires the official typed element API"
#endif

#if !defined(PTO_TILEOP_API_HAS_DENSE_F16_U16_ELEMENT_TILE) ||               \
    PTO_TILEOP_API_HAS_DENSE_F16_U16_ELEMENT_TILE != 1
#error "concat_gather_f16_original requires the official dense F16 element API"
#endif

#include "benchmark.h"
#include "linx_print.h"
#include "basic_op/concat/detail/concat_gather.hpp"

#include <cstddef>
#include <cstdint>

#ifndef CONCAT_GATHER_F16_ALIAS_PROBE
#define CONCAT_GATHER_F16_ALIAS_PROBE 0
#endif

namespace concat_gather_f16_original_benchmark {

constexpr std::size_t kMaxDimensions = 8;
constexpr std::size_t kDataDimensions = 2;
constexpr std::size_t kConcatDimension = 1;
constexpr int kInputElements = 4000 * 256 * 8;
constexpr int kUsedInputElements = 1000 * 256 * 8;
constexpr bool kAliasProbe = CONCAT_GATHER_F16_ALIAS_PROBE == 1;
constexpr int kOutputElements = kAliasProbe ? 512 : 256 * 8000;
constexpr int kTileElements = 512;
constexpr std::size_t kGuardElements = 32;
constexpr std::size_t kAliasOutputOffset = 2049;
constexpr uint16_t kGuardBits = UINT16_C(0x5a5a);

// Preserve the legacy compile.all sizes and shapes. Its automatic buffers
// were uninitialized and its file read was disabled, so use a deterministic
// raw-bit fixture here to make the original configuration verifiable.
inline uint16_t input_bits(std::size_t element) {
  return static_cast<uint16_t>(
      (static_cast<uint32_t>(element) * 40503u + 0x35a7u) & 0xffffu);
}

inline void store_bits(__half &destination, uint16_t bits) {
  __builtin_memcpy(&destination, &bits, sizeof(bits));
}

inline uint16_t load_bits(const __half &source) {
  uint16_t bits;
  __builtin_memcpy(&bits, &source, sizeof(bits));
  return bits;
}

inline uint32_t source_element(uint32_t global) {
  const uint32_t row = (global / 8000u) % 256u;
  const uint32_t col = global % 8000u;
  const uint32_t input_no = col / 8u;
  return input_no * (256u * 8u) + row * 8u + (col % 8u);
}

} // namespace concat_gather_f16_original_benchmark

using namespace concat_gather_f16_original_benchmark;

#ifdef CONCAT_GATHER_F16_LINKED_INPUTS
#define CGF16_DATA_DECL extern
#else
#define CGF16_DATA_DECL
#endif
extern "C" {
alignas(4096) CGF16_DATA_DECL __half concat_gather_f16_original_input[kInputElements];
alignas(64) CGF16_DATA_DECL __half
    concat_gather_f16_original_output_guard_before[kGuardElements];
alignas(64) CGF16_DATA_DECL __half concat_gather_f16_original_output[kOutputElements];
alignas(64) CGF16_DATA_DECL __half
    concat_gather_f16_original_output_guard_after[kGuardElements];
alignas(32) CGF16_DATA_DECL std::size_t
    concat_gather_f16_original_in_shape[kMaxDimensions];
alignas(32) CGF16_DATA_DECL std::size_t
    concat_gather_f16_original_out_shape[kMaxDimensions];
alignas(32) uint32_t concat_gather_f16_original_status[14];
}
#undef CGF16_DATA_DECL

#ifdef CONCAT_GATHER_F16_LINKED_INPUTS
#include "concat_gather_f16_input.inc"
#endif

int main() {
#ifndef CONCAT_GATHER_F16_LINKED_INPUTS
  for (std::size_t element = 0; element < kInputElements; ++element) {
    store_bits(concat_gather_f16_original_input[element], input_bits(element));
  }
  for (std::size_t element = 0; element < kOutputElements; ++element) {
    store_bits(concat_gather_f16_original_output[element], kGuardBits);
  }
  for (std::size_t element = 0; element < kGuardElements; ++element) {
    store_bits(concat_gather_f16_original_output_guard_before[element],
               kGuardBits);
    store_bits(concat_gather_f16_original_output_guard_after[element],
               kGuardBits);
  }
  for (std::size_t dimension = 0; dimension < kMaxDimensions; ++dimension) {
    concat_gather_f16_original_in_shape[dimension] = 0;
    concat_gather_f16_original_out_shape[dimension] = 0;
  }
  concat_gather_f16_original_in_shape[0] = 256;
  concat_gather_f16_original_in_shape[1] = 8;
  concat_gather_f16_original_out_shape[0] = 256;
  concat_gather_f16_original_out_shape[1] = 8000;
#endif

  BENCHSTART;
  if constexpr (kAliasProbe) {
    [[clang::noinline]]
    concat_gather<__half, kMaxDimensions, kInputElements, kOutputElements,
                  kTileElements, kDataDimensions, kConcatDimension>(
        concat_gather_f16_original_input,
        concat_gather_f16_original_input + kAliasOutputOffset,
        concat_gather_f16_original_in_shape,
        concat_gather_f16_original_out_shape);
  } else {
    [[clang::noinline]]
    concat_gather<__half, kMaxDimensions, kInputElements, kOutputElements,
                  kTileElements, kDataDimensions, kConcatDimension>(
        concat_gather_f16_original_input,
        concat_gather_f16_original_output,
        concat_gather_f16_original_in_shape,
        concat_gather_f16_original_out_shape);
  }
  BENCHEND;

#ifdef CONCAT_GATHER_F16_LINKED_INPUTS
  // This mode records call completion only. The suite's host-side oracle
  // compares the full buffers, including all unused input, guards and shapes;
  // a zero return/status alone does not establish correctness.
  const uint32_t output_failures = 0;
  const uint32_t guard_failures = 0;
  const uint32_t input_failures = 0;
  const uint32_t shape_failures = 0;
#else
  uint32_t output_failures = 0;
  for (uint32_t global = 0; global < kOutputElements; ++global) {
    const uint16_t expected = input_bits(source_element(global));
    const uint16_t actual = kAliasProbe
        ? load_bits(concat_gather_f16_original_input
                        [kAliasOutputOffset + global])
        : load_bits(concat_gather_f16_original_output[global]);
    output_failures += actual != expected;
  }

  uint32_t guard_failures = 0;
  for (std::size_t element = 0; element < kGuardElements; ++element) {
    guard_failures += load_bits(
                          concat_gather_f16_original_output_guard_before
                              [element]) != kGuardBits;
    guard_failures += load_bits(
                          concat_gather_f16_original_output_guard_after
                              [element]) != kGuardBits;
  }
  if constexpr (kAliasProbe) {
    for (std::size_t element = 0; element < kOutputElements; ++element) {
      guard_failures +=
          load_bits(concat_gather_f16_original_output[element]) != kGuardBits;
    }
  }

  uint32_t input_failures = 0;
  for (std::size_t element = 0; element < kInputElements; ++element) {
    uint16_t expected = input_bits(element);
    if constexpr (kAliasProbe) {
      if (element >= kAliasOutputOffset &&
          element < kAliasOutputOffset + kOutputElements) {
        const uint32_t global =
            static_cast<uint32_t>(element - kAliasOutputOffset);
        expected = input_bits(source_element(global));
      }
    }
    input_failures +=
        load_bits(concat_gather_f16_original_input[element]) != expected;
  }

  uint32_t shape_failures = 0;
  shape_failures += concat_gather_f16_original_in_shape[0] != 256;
  shape_failures += concat_gather_f16_original_in_shape[1] != 8;
  shape_failures += concat_gather_f16_original_out_shape[0] != 256;
  shape_failures += concat_gather_f16_original_out_shape[1] != 8000;
  for (std::size_t dimension = 2; dimension < kMaxDimensions; ++dimension) {
    shape_failures += concat_gather_f16_original_in_shape[dimension] != 0;
    shape_failures += concat_gather_f16_original_out_shape[dimension] != 0;
  }
#endif

  const uint32_t failures = output_failures + guard_failures + input_failures +
                            shape_failures;
  concat_gather_f16_original_status[0] = failures;
  concat_gather_f16_original_status[1] = output_failures;
  concat_gather_f16_original_status[2] = guard_failures;
  concat_gather_f16_original_status[3] = input_failures;
  concat_gather_f16_original_status[4] = shape_failures;
  concat_gather_f16_original_status[5] = kInputElements;
  concat_gather_f16_original_status[6] = kOutputElements;
  concat_gather_f16_original_status[7] = kTileElements;
  concat_gather_f16_original_status[8] = 256;
  concat_gather_f16_original_status[9] = 8;
  concat_gather_f16_original_status[10] = 8000;
  concat_gather_f16_original_status[11] = 1000;
  concat_gather_f16_original_status[12] = kUsedInputElements;
  concat_gather_f16_original_status[13] = 0x43474631u; // CGF1

  linxi_puts("=== concat_gather_f16_original ===");
  linxi_put_kv("failures", failures);
  return failures == 0 ? 0 : 1;
}
