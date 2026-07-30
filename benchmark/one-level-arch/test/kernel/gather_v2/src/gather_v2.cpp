#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "gather_v2/gather_v2.hpp"

#ifndef GATHER_DTYPE
#define GATHER_DTYPE float
#endif

#ifndef GATHER_ITYPE
#define GATHER_ITYPE std::uint32_t
#endif

#ifndef RANKs
#define RANKs 2
#endif

#ifndef GATHER_DIMs
#define GATHER_DIMs 0
#endif

#ifndef INPUT_ELEMENTSs
#define INPUT_ELEMENTSs 160
#endif

#ifndef OUTPUT_ELEMENTSs
#define OUTPUT_ELEMENTSs 96
#endif

#ifndef INDEX_ELEMENTSs
#define INDEX_ELEMENTSs 6
#endif

#ifndef TILE_ELEMENTSs
#define TILE_ELEMENTSs 32
#endif

#ifndef INPUT_SHAPEs
#define INPUT_SHAPEs 10, 16
#endif

#ifndef OUTPUT_SHAPEs
#define OUTPUT_SHAPEs 6, 16
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
  using dtype = GATHER_DTYPE;
  using itype = GATHER_ITYPE;

  static_assert(RANKs > 0, "Rank must be positive");
  static_assert(GATHER_DIMs >= 0 && GATHER_DIMs < RANKs,
                "GatherDim must be in [0, Rank)");
  static_assert(INPUT_ELEMENTSs > 0, "InputElements must be positive");
  static_assert(OUTPUT_ELEMENTSs > 0, "OutputElements must be positive");
  static_assert(INDEX_ELEMENTSs > 0, "IndexElements must be positive");

  std::uint32_t input_shape[RANKs] = {INPUT_SHAPEs};
  std::uint32_t output_shape[RANKs] = {OUTPUT_SHAPEs};

  // The Row specialization uses a 128-byte physical index tile with one valid
  // element. Padding the backing buffers also keeps their final physical
  // transaction inside allocated storage.
  constexpr int kIndexPaddingElements =
      128 / static_cast<int>(sizeof(itype));
  constexpr int kDataPaddingElements = TILE_ELEMENTSs;
  alignas(32) dtype input[INPUT_ELEMENTSs + kDataPaddingElements] = {};
  alignas(32) itype index[INDEX_ELEMENTSs + kIndexPaddingElements] = {};
  alignas(32) dtype output[OUTPUT_ELEMENTSs + kDataPaddingElements];
  alignas(32) dtype reference[OUTPUT_ELEMENTSs];

  std::uint64_t input_shape_elements = 1;
  std::uint64_t output_shape_elements = 1;
  for (int dim = 0; dim < RANKs; ++dim) {
    if (input_shape[dim] == 0 || output_shape[dim] == 0) {
      if (get_thread_idx() == 0) {
        printf("FAIL: shape[%d] must be positive\n", dim);
      }
      return 2;
    }
    if (dim != GATHER_DIMs && input_shape[dim] != output_shape[dim]) {
      if (get_thread_idx() == 0) {
        printf("FAIL: non-gather dimension %d differs (%u vs %u)\n", dim,
               input_shape[dim], output_shape[dim]);
      }
      return 2;
    }
    input_shape_elements *= input_shape[dim];
    output_shape_elements *= output_shape[dim];
  }

  if (input_shape_elements != static_cast<std::uint64_t>(INPUT_ELEMENTSs) ||
      output_shape_elements != static_cast<std::uint64_t>(OUTPUT_ELEMENTSs) ||
      output_shape[GATHER_DIMs] != INDEX_ELEMENTSs) {
    if (get_thread_idx() == 0) {
      printf("FAIL: inconsistent test shape or element count\n");
    }
    return 2;
  }

  for (int i = 0; i < INPUT_ELEMENTSs; ++i) {
    const float value =
        static_cast<float>((i * 29 + 13) % 211 - 105) * 0.0625f;
    input[i] = static_cast<dtype>(value);
  }

  for (int i = 0; i < INDEX_ELEMENTSs; ++i) {
    const std::uint32_t selected =
        (static_cast<std::uint32_t>(i) * 7U + 3U) %
        input_shape[GATHER_DIMs];
    index[i] = static_cast<itype>(selected);
  }

  const dtype sentinel = static_cast<dtype>(-17.0f);
  for (int i = 0; i < OUTPUT_ELEMENTSs + kDataPaddingElements; ++i) {
    output[i] = sentinel;
  }
  for (int i = 0; i < OUTPUT_ELEMENTSs; ++i) {
    reference[i] = sentinel;
  }

  // Scalar reference for a one-dimensional index:
  // output[..., i, ...] = input[..., index[i], ...].
  for (std::uint32_t linear = 0; linear < OUTPUT_ELEMENTSs; ++linear) {
    std::uint32_t coordinate[RANKs] = {};
    std::uint64_t logical_stride = 1;
    for (int dim = RANKs - 1; dim >= 0; --dim) {
      coordinate[dim] =
          static_cast<std::uint32_t>((linear / logical_stride) %
                                     output_shape[dim]);
      logical_stride *= output_shape[dim];
    }

    std::uint64_t input_linear = 0;
    for (int dim = 0; dim < RANKs; ++dim) {
      std::uint32_t input_coordinate = coordinate[dim];
      if (dim == GATHER_DIMs) {
        input_coordinate =
            static_cast<std::uint32_t>(index[coordinate[dim]]);
      }
      input_linear = input_linear * input_shape[dim] + input_coordinate;
    }
    reference[linear] = input[input_linear];
  }

  supernpu::tile_isa::gather_v2<
      dtype, itype, RANKs, GATHER_DIMs, INPUT_ELEMENTSs, OUTPUT_ELEMENTSs,
      TILE_ELEMENTSs>(input, index, output, input_shape, output_shape);

  // bstart.std executes the program on four PEs. Only PE0 reports the shared
  // result after the SPMD kernel call has completed.
  if (get_thread_idx() != 0) {
    return 0;
  }

  int mismatch_count = 0;
  for (int i = 0; i < OUTPUT_ELEMENTSs; ++i) {
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
    printf("FAIL: gather_v2 found %d mismatches\n", mismatch_count);
    return 1;
  }

  printf("PASS: gather_v2 checked %d output elements (rank=%d, dim=%d)\n",
         OUTPUT_ELEMENTSs, RANKs, GATHER_DIMs);
  return 0;
}
