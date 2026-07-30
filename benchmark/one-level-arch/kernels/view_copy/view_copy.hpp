/**
 * @file view_copy.hpp
 * @brief Copy between strided tensor views with Tile ISA gather/scatter.
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
template <typename DType, int Rank, int TileElements, int ValidElements,
          typename InputGlobal, typename OutputGlobal>
inline void process_view_copy_tile(
    InputGlobal &input_global, OutputGlobal &output_global,
    std::uint32_t linear_base, const std::uint32_t *shape,
    const std::uint32_t *input_global_stride,
    const std::uint32_t *output_global_stride) {
  static_assert(ValidElements > 0, "ValidElements must be positive");
  static_assert(ValidElements <= TileElements,
                "ValidElements cannot exceed the physical tile size");

  using DataTile =
      pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                pto::BLayout::RowMajor, 1, ValidElements>;
  using OffsetTile =
      pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                pto::BLayout::RowMajor, 1, ValidElements>;

  DataTile output_tile;
  OffsetTile input_offset_tile;
  OffsetTile output_offset_tile;

  OffsetTile linear_index;
  OffsetTile quotient;
  OffsetTile cycle;
  OffsetTile cycle_base;
  OffsetTile coordinate;
  OffsetTile input_contribution;
  OffsetTile output_contribution;

  TCI(linear_index, linear_base);
  TEXPANDS(input_offset_tile, static_cast<std::uint32_t>(0));
  TEXPANDS(output_offset_tile, static_cast<std::uint32_t>(0));

  std::uint32_t logical_stride = 1;
  for (int dim = Rank - 1; dim >= 0; --dim) {
    if (logical_stride == 1) {
      TCVT(quotient, linear_index);
    } else {
      TDIVS(quotient, linear_index, logical_stride);
    }

    // coordinate = (linear_index / logical_stride) % shape[dim]
    TDIVS(cycle, quotient, shape[dim]);
    TMULS(cycle_base, cycle, shape[dim]);
    TSUB(coordinate, quotient, cycle_base);

    // MGATHER/MSCATTER offsets use byte units.
    TMULS(input_contribution, coordinate,
          input_global_stride[dim] *
              static_cast<std::uint32_t>(sizeof(DType)));
    TMULS(output_contribution, coordinate,
          output_global_stride[dim] *
              static_cast<std::uint32_t>(sizeof(DType)));

    TADD(input_offset_tile, input_offset_tile, input_contribution);
    TADD(output_offset_tile, output_offset_tile, output_contribution);

    logical_stride *= shape[dim];
  }

  MGATHER(output_tile, input_global, input_offset_tile);
  MSCATTER(output_global, output_tile, output_offset_tile);
}

} // namespace detail

/**
 * Copy Elements logical elements between two strided views of the same shape.
 *
 * input_offset/output_offset and gather/scatter offsets are expressed in bytes.
 * input_global_stride/output_global_stride are expressed in elements. The
 * caller must ensure that offsets are aligned to DType, product(shape) equals
 * Elements, and both strided storage spans fit in uint32 byte offsets.
 *
 * ThreadNum defaults to one for the regular single-thread test suite. A
 * multi-thread caller must instantiate the kernel with its launch thread count.
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

  using InputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
  using OutputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, Elements>>;

  InputGlobal input_global(input + input_offset / sizeof(DType));
  OutputGlobal output_global(output + output_offset / sizeof(DType));

  constexpr std::uint32_t ThreadNum = 4;
  std::uint32_t thread_id = get_thread_idx();

  for (int tile = static_cast<int>(thread_id); tile < kFullTiles;
       tile += ThreadNum) {
    detail::process_view_copy_tile<DType, Rank, TileElements, TileElements>(
        input_global, output_global,
        static_cast<std::uint32_t>(tile * TileElements), shape,
        input_global_stride, output_global_stride);
  }

  if constexpr (kTailElements != 0) {
    constexpr int kTailOwner = kFullTiles % ThreadNum;
    if (thread_id == static_cast<std::uint32_t>(kTailOwner)) {
      detail::process_view_copy_tile<DType, Rank, TileElements, kTailElements>(
          input_global, output_global,
          static_cast<std::uint32_t>(kFullTiles * TileElements), shape,
          input_global_stride, output_global_stride);
    }
  }
}

} // namespace supernpu::tile_isa

#endif // SUPERNPU_VIEW_COPY_HPP
