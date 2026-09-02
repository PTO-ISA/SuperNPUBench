#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "view_copy/view_copy_pto.hpp"

#ifndef DType
#define DType int32_t
#endif

#ifndef RANKs
#define RANKs 3
#endif

#ifndef ELEMENTSs
#define ELEMENTSs 64
#endif

#ifndef TILE_ELEMENTSs
#define TILE_ELEMENTSs 32
#endif

#ifndef SHAPEs
#define SHAPEs 2, 4, 8
#endif

#ifndef INPUT_GLOBAL_STRIDEs
#define INPUT_GLOBAL_STRIDEs 32, 1, 4
#endif

#ifndef OUTPUT_GLOBAL_STRIDEs
#define OUTPUT_GLOBAL_STRIDEs 32, 8, 1
#endif

#ifndef INPUT_OFFSET_BYTESs
#define INPUT_OFFSET_BYTESs 32
#endif

#ifndef OUTPUT_OFFSET_BYTESs
#define OUTPUT_OFFSET_BYTESs 64
#endif

#ifndef INPUT_STORAGE_ELEMENTSs
#define INPUT_STORAGE_ELEMENTSs ELEMENTSs
#endif

#ifndef OUTPUT_STORAGE_ELEMENTSs
#define OUTPUT_STORAGE_ELEMENTSs ELEMENTSs
#endif

#ifndef ABS_TOLs
#define ABS_TOLs 0.001f
#endif

#ifndef REL_TOLs
#define REL_TOLs 0.001f
#endif

static float absolute_value(float value) {
  return value < 0.0f ? -value : value;
}

int main() {
  using dtype = DType;

  static_assert(INPUT_OFFSET_BYTESs % sizeof(dtype) == 0,
                "The input byte offset must be aligned to DType");
  static_assert(OUTPUT_OFFSET_BYTESs % sizeof(dtype) == 0,
                "The output byte offset must be aligned to DType");
  static_assert(RANKs > 0, "Rank must be positive");
  static_assert(INPUT_STORAGE_ELEMENTSs > 0,
                "Input storage must contain at least one element");
  static_assert(OUTPUT_STORAGE_ELEMENTSs > 0,
                "Output storage must contain at least one element");

  constexpr int kInputOffsetElements = INPUT_OFFSET_BYTESs / sizeof(dtype);
  constexpr int kOutputOffsetElements = OUTPUT_OFFSET_BYTESs / sizeof(dtype);
  constexpr int kInputBufferElements =
      INPUT_STORAGE_ELEMENTSs + kInputOffsetElements;
  constexpr int kOutputBufferElements =
      OUTPUT_STORAGE_ELEMENTSs + kOutputOffsetElements;

  std::uint32_t shape[RANKs] = {SHAPEs};
  std::uint32_t input_global_stride[RANKs] = {INPUT_GLOBAL_STRIDEs};
  std::uint32_t output_global_stride[RANKs] = {OUTPUT_GLOBAL_STRIDEs};

  alignas(32) dtype input[kInputBufferElements] = {};
  alignas(32) dtype output[kOutputBufferElements];
  alignas(32) dtype reference[kOutputBufferElements];

  const dtype sentinel = static_cast<dtype>(-7.0f);
  for (int i = 0; i < kOutputBufferElements; ++i) {
    output[i] = sentinel;
    reference[i] = sentinel;
  }

  // Populate physical input storage with deterministic, non-zero values. Using
  // the physical index makes an incorrect input stride visible in the result.
  for (int i = 0; i < INPUT_STORAGE_ELEMENTSs; ++i) {
    const float value =
        static_cast<float>((i * 17 + 11) % 101 - 50) * 0.125f;
    input[kInputOffsetElements + i] = static_cast<dtype>(value);
  }

  std::uint64_t shape_elements = 1;
  for (int dim = 0; dim < RANKs; ++dim) {
    if (shape[dim] == 0) {
      printf("FAIL: shape[%d] is zero\n", dim);
      return 2;
    }
    shape_elements *= shape[dim];
  }
  if (shape_elements != static_cast<std::uint64_t>(ELEMENTSs)) {
    printf("FAIL: product(shape)=%llu, Elements=%d\n",
           static_cast<unsigned long long>(shape_elements), ELEMENTSs);
    return 2;
  }

  // Scalar reference: map every logical coordinate through the input and
  // output global strides. The strides are expressed in elements.
  for (std::uint32_t linear = 0; linear < ELEMENTSs; ++linear) {
    std::uint64_t input_index = 0;
    std::uint64_t output_index = 0;
    std::uint64_t logical_stride = 1;

    for (int dim = RANKs - 1; dim >= 0; --dim) {
      const std::uint64_t coordinate =
          (linear / logical_stride) % shape[dim];
      input_index += coordinate * input_global_stride[dim];
      output_index += coordinate * output_global_stride[dim];
      logical_stride *= shape[dim];
    }

    if (input_index >= INPUT_STORAGE_ELEMENTSs ||
        output_index >= OUTPUT_STORAGE_ELEMENTSs) {
      printf("FAIL: reference index out of range at logical element %u "
             "(input=%llu, output=%llu)\n",
             linear, static_cast<unsigned long long>(input_index),
             static_cast<unsigned long long>(output_index));
      return 2;
    }

    reference[kOutputOffsetElements + output_index] =
        input[kInputOffsetElements + input_index];
  }

  supernpu::tile_isa::tile_view_copy<dtype, RANKs, ELEMENTSs, TILE_ELEMENTSs>(
      input, output, shape, INPUT_OFFSET_BYTESs, OUTPUT_OFFSET_BYTESs,
      input_global_stride, output_global_stride);

  int mismatch_count = 0;
  for (int i = 0; i < kOutputBufferElements; ++i) {
    const float actual = static_cast<float>(output[i]);
    const float expected = static_cast<float>(reference[i]);
    const float difference = absolute_value(actual - expected);
    const float tolerance =
        static_cast<float>(ABS_TOLs) +
        static_cast<float>(REL_TOLs) * absolute_value(expected);

    const bool has_nan = actual != actual || expected != expected;
    if (has_nan || difference > tolerance) {
      if (mismatch_count < 8) {
        printf("Mismatch[%d]: actual=%f expected=%f diff=%f tol=%f\n", i,
               actual, expected, difference, tolerance);
      }
      ++mismatch_count;
    }
  }

  if (mismatch_count != 0) {
    printf("FAIL: view_copy found %d mismatches\n", mismatch_count);
    return 1;
  }

  printf("PASS: view_copy checked %d logical elements\n", ELEMENTSs);
  return 0;
}
