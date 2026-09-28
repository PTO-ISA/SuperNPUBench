/** Gather a shared one-dimensional index vector along an arbitrary axis. */
#ifndef SUPERNPU_GATHER_V2_HPP
#define SUPERNPU_GATHER_V2_HPP

#include "solution/common/copy_tile.hpp"
#include <cstdint>

namespace supernpu::tile_isa {
namespace detail {

template <typename DType, typename IType, int Rank, int GatherDim, int TileElements, int ValidElements,
          typename InputGlobal, typename IndexGlobal, typename OutputGlobal>
inline void process_gather_tile(
    InputGlobal &input_global, IndexGlobal &index_global, OutputGlobal &output_global,
    std::uint32_t linear_base, const std::uint32_t *input_shape, const std::uint32_t *output_shape
)
{
  static_assert(ValidElements > 0, "ValidElements must be positive");
  static_assert(ValidElements <= TileElements,
                "ValidElements cannot exceed the physical tile size");

  using DataTile =
      pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                pto::BLayout::RowMajor, 1, ValidElements>;
  using OffsetTile =
      pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                pto::BLayout::RowMajor, 1, ValidElements>;
  using IndexTile =
      pto::Tile<pto::Location::Vec, IType, 1, TileElements,
                pto::BLayout::RowMajor, 1, ValidElements>;

  DataTile output_tile;
  OffsetTile input_offset_tile;

  OffsetTile linear_index;
  OffsetTile quotient;
  OffsetTile cycle;
  OffsetTile cycle_base;
  OffsetTile coordinate;
  OffsetTile contribution;

  TCI(linear_index, linear_base);
  TEXPANDS(input_offset_tile, static_cast<std::uint32_t>(0));

  std::uint32_t output_stride = 1;
  std::uint32_t input_stride = 1;
  for (int dim = Rank - 1; dim >= 0; --dim) {
    if (output_stride == 1) {
      TCVT(quotient, linear_index);
    } else {
      TDIVS(quotient, linear_index, output_stride);
    }

    // coordinate = (linear_index / output_stride) % output_shape[dim]
    TDIVS(cycle, quotient, output_shape[dim]);
    TMULS(cycle_base, cycle, output_shape[dim]);
    TSUB(coordinate, quotient, cycle_base);

    // transfer logical coordinate to gather coordinate
    if (dim == GatherDim) {
      OffsetTile index_offset;
      IndexTile index_tile;
      // Indexed TLSU consumes byte displacements (PTO 0.58.4+).
      TMULS(index_offset, coordinate, static_cast<std::uint32_t>(sizeof(IType)));
      MGATHER(index_tile, index_global, index_offset);
      TCVT(coordinate, index_tile);
    }

    TMULS(contribution, coordinate,
          input_stride * static_cast<std::uint32_t>(sizeof(DType)));

    TADD(input_offset_tile, input_offset_tile, contribution);

    output_stride *= output_shape[dim];
    input_stride *= input_shape[dim];
  }

  MGATHER(output_tile, input_global, input_offset_tile);
  OutputGlobal tile_output_global(output_global.data() + linear_base);
  TSTORE(tile_output_global, output_tile);
}

template <typename DType, typename IType, int Rank, int GatherDim,
          int InputElements, int OutputElements, int TileElements>
__attribute__((always_inline)) inline void gather_indexed(
    DType *input, IType *index, DType *output,
    const std::uint32_t *input_shape, const std::uint32_t *output_shape) {
  constexpr int kFullTiles = OutputElements / TileElements;
  constexpr int kTailElements = OutputElements % TileElements;

  using InputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, InputElements>>;
  using IndexGlobal =
      pto::global_tensor<IType, pto::RowMajor<1, OutputElements>>;
  using OutputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, OutputElements>>;

  InputGlobal input_global(input);
  IndexGlobal index_global(index);
  OutputGlobal output_global(output);

  constexpr std::uint32_t ThreadNum = 4;
  std::uint32_t thread_id = get_thread_idx();

  for (int tile = static_cast<int>(thread_id); tile < kFullTiles;
       tile += ThreadNum) {
    detail::process_gather_tile<DType, IType, Rank, GatherDim, TileElements, TileElements>(
        input_global, index_global, output_global,
        static_cast<std::uint32_t>(tile * TileElements), input_shape, output_shape);
  }

  if constexpr (kTailElements != 0) {
    constexpr int kTailOwner = kFullTiles % ThreadNum;
    if (thread_id == static_cast<std::uint32_t>(kTailOwner)) {
      detail::process_gather_tile<DType, IType, Rank, GatherDim, TileElements, kTailElements>(
          input_global, index_global, output_global,
          static_cast<std::uint32_t>(kFullTiles * TileElements), input_shape, output_shape);
    }
  }
}

} // namespace detail

// Input and output are contiguous, and their shapes agree outside GatherDim.
// Indices must be valid coordinates in input_shape[GatherDim]. All four PEs
// participate. Input/index and output must not overlap, as in the original
// parallel gather implementation.
template <typename DType, typename IType, int Rank, int GatherDim,
          int InputElements, int OutputElements, int TileElements = 512>
void gather_v2(DType *input, IType *index, DType *output,
               const std::uint32_t *input_shape,
               const std::uint32_t *output_shape) {
  static_assert(Rank > 0 && GatherDim >= 0 && GatherDim < Rank);
  static_assert(InputElements > 0 && OutputElements > 0 && TileElements > 0);
  static_assert(TileElements * sizeof(DType) >= 128);
  static_assert(TileElements * sizeof(DType) % 32 == 0);
  static_assert(TileElements * sizeof(std::uint32_t) >= 128);
  static_assert(TileElements * sizeof(std::uint32_t) % 32 == 0);

  // For at most one Tile per PE, avoid runtime row-path selection and
  // its setup; the flattened indexed path already balances the whole job.
  if constexpr (OutputElements <= 4 * TileElements) {
    detail::gather_indexed<DType, IType, Rank, GatherDim,
                          InputElements, OutputElements, TileElements>(
        input, index, output, input_shape, output_shape);
  } else {
    constexpr std::uint32_t threads = 4;
    const std::uint32_t tid = get_thread_idx();
    std::uint32_t inner = 1;
    for (int dim = GatherDim + 1; dim < Rank; ++dim)
      inner *= input_shape[dim];
    const std::uint32_t count = output_shape[GatherDim];
    const std::uint32_t inputCount = input_shape[GatherDim];

    if (inner >= 128 / sizeof(DType)) {
      // [outer, axis, inner]: each selected row is one contiguous segment.
      // Assign whole rows when there are enough, so the index is read once per
      // row. With fewer than four rows, stripe chunks to use all four PEs.
      const std::uint32_t rows = OutputElements / inner;
      const bool stripe = rows < threads;
      if (stripe && tid * TileElements >= inner) return;
      for (std::uint32_t row = stripe ? 0 : tid; row < rows;
           row += stripe ? 1 : threads) {
        const std::uint32_t outer = row / count;
        const std::uint32_t selected = static_cast<std::uint32_t>(index[row % count]);
        DType *src = input + (outer * inputCount + selected) * inner;
        DType *dst = output + row * inner;
        for (std::uint32_t col = stripe ? tid * TileElements : 0;
             col < inner; col += (stripe ? threads : 1) * TileElements) {
          const std::uint32_t valid = inner - col < TileElements
                                        ? inner - col : TileElements;
          detail::copy_contiguous_tile<DType, TileElements>(src + col, dst + col, valid);
        }
      }
    } else {
      // Narrow rows pack better into flattened indexed Tiles.
      detail::gather_indexed<DType, IType, Rank, GatherDim,
                            InputElements, OutputElements, TileElements>(
          input, index, output, input_shape, output_shape);
    }
  }
}

} // namespace supernpu::tile_isa
#endif // SUPERNPU_GATHER_V2_HPP
