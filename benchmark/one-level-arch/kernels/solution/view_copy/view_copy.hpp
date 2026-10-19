/**
 * @file view_copy.hpp
 * @brief Copy between strided tensor views with element-wise Tile lowering.
 */

#ifndef SUPERNPU_VIEW_COPY_HPP
#define SUPERNPU_VIEW_COPY_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace supernpu::tile_isa {
namespace detail {

/**
 * Process one physical tile. ValidElements is TileElements for a full tile and
 * kTailElements for the final partial tile. Keeping ValidElements as a template
 * parameter lets the PTO tile valid region remain a compile-time constant.
 */
template <typename DType, int Rank, int TileElements, int ValidElements>
inline void process_view_copy_tile(
    const DType *input, DType *output, std::uint32_t linear_base,
    const std::uint32_t *shape,
    const std::uint32_t *input_global_stride,
    const std::uint32_t *output_global_stride) {
  static_assert(ValidElements > 0, "ValidElements must be positive");
  static_assert(ValidElements <= TileElements,
                "ValidElements cannot exceed the physical tile size");

  std::uint32_t local_shape[Rank];
  std::uint32_t local_input_stride[Rank];
  std::uint32_t local_output_stride[Rank];
  for (int dim = 0; dim < Rank; ++dim) {
    local_shape[dim] = shape[dim];
    local_input_stride[dim] = input_global_stride[dim];
    local_output_stride[dim] = output_global_stride[dim];
  }

  pto::ElementTile<DType, TileElements> snapshot;
  auto &snapshot_elements = pto::TPARTELEMENT(snapshot);

  // Finish every source read before this tile performs any destination write.
  // Later tiles intentionally observe writes from earlier tiles when the views
  // alias, matching the original per-tile MGATHER/MSCATTER behavior.
#pragma pto element for
  for (std::uint32_t element = 0; element < TileElements; ++element) {
    if (element < ValidElements) {
      const std::uint32_t linear = linear_base + element;
      std::uint32_t input_index = 0;
      std::uint32_t logical_stride = 1;
      for (int dim = Rank - 1; dim >= 0; --dim) {
        const std::uint32_t coordinate =
            (linear / logical_stride) % local_shape[dim];
        input_index += coordinate * local_input_stride[dim];
        logical_stride *= local_shape[dim];
      }
      snapshot_elements[element] = input[input_index];
    }
  }

  // Keep the source snapshot as a Tile transport between the read and write
  // regions. This makes the per-tile alias contract explicit to lowering.
  pto::ElementTile<DType, TileElements> transported;
  TMOV(transported, snapshot);
  auto &transported_elements = pto::TPARTELEMENT(transported);

#pragma pto element for
  for (std::uint32_t element = 0; element < TileElements; ++element) {
    if (element < ValidElements) {
      const std::uint32_t linear = linear_base + element;
      std::uint32_t output_index = 0;
      std::uint32_t logical_stride = 1;
      for (int dim = Rank - 1; dim >= 0; --dim) {
        const std::uint32_t coordinate =
            (linear / logical_stride) % local_shape[dim];
        output_index += coordinate * local_output_stride[dim];
        logical_stride *= local_shape[dim];
      }
      output[output_index] = transported_elements[element];
    }
  }
}

} // namespace detail

/**
 * Copy Elements logical elements between two strided views of the same shape.
 *
 * input_offset/output_offset are expressed in bytes; gather/scatter indices and
 * input_global_stride/output_global_stride are expressed in elements. The
 * caller must ensure that offsets are aligned to DType, product(shape) equals
 * Elements, and both strided storage spans fit in uint32 byte offsets.
 *
 * The caller must launch all four PEs and synchronize their shared buffers.
 */
template <typename DType, int Rank, int Elements, int TileElements = 512>
void tile_view_copy(DType *input, DType *output, const std::uint32_t *shape,
                    std::uint32_t input_offset,
                    std::uint32_t output_offset,
                    const std::uint32_t *input_global_stride,
                    const std::uint32_t *output_global_stride) {
  static_assert(Rank > 0, "Rank must be positive");
  static_assert(Elements > 0, "Elements must be positive");
  static_assert(TileElements > 0, "TileElements must be positive");
  static_assert(TileElements * static_cast<int>(sizeof(DType)) >= 128,
                "The physical data tile must be at least 128 bytes");
  static_assert(TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                "A RowMajor tile row must be 32-byte aligned");
  static_assert(
      TileElements * static_cast<int>(sizeof(std::uint32_t)) >= 128,
      "The physical offset tile must be at least 128 bytes");
  static_assert(
      TileElements * static_cast<int>(sizeof(std::uint32_t)) % 32 == 0,
      "An offset tile row must be 32-byte aligned");

  constexpr int kFullTiles = Elements / TileElements;
  constexpr int kTailElements = Elements % TileElements;

  const DType *input_view = input + input_offset / sizeof(DType);
  DType *output_view = output + output_offset / sizeof(DType);

  constexpr std::uint32_t ThreadNum = 4;
  std::uint32_t thread_id = get_thread_idx();

  for (int tile = static_cast<int>(thread_id); tile < kFullTiles;
       tile += ThreadNum) {
    detail::process_view_copy_tile<DType, Rank, TileElements, TileElements>(
        input_view, output_view,
        static_cast<std::uint32_t>(tile * TileElements), shape,
        input_global_stride, output_global_stride);
  }

  if constexpr (kTailElements != 0) {
    constexpr int kTailOwner = kFullTiles % ThreadNum;
    if (thread_id == static_cast<std::uint32_t>(kTailOwner)) {
      detail::process_view_copy_tile<DType, Rank, TileElements, kTailElements>(
          input_view, output_view,
          static_cast<std::uint32_t>(kFullTiles * TileElements), shape,
          input_global_stride, output_global_stride);
    }
  }
}

} // namespace supernpu::tile_isa

#endif // SUPERNPU_VIEW_COPY_HPP
