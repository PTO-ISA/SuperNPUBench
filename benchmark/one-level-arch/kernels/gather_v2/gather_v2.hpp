/**
 * @file gather_v2.hpp
 * @brief Copy between strided tensor views with Tile ISA gather/scatter.
 */

#ifndef SUPERNPU_GATHER_V2_HPP
#define SUPERNPU_GATHER_V2_HPP

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Preserve the rank-2/axis-0 Coalesce::Row implementation while the current
// compiler issue is being fixed. Set this macro to 1 to re-enable the path.
#ifndef SUPERNPU_ENABLE_GATHER_V2_COALESCE_ROW
#define SUPERNPU_ENABLE_GATHER_V2_COALESCE_ROW 0
#endif

namespace supernpu::tile_isa {
namespace detail {

#if SUPERNPU_ENABLE_GATHER_V2_COALESCE_ROW
/**
 * Specialized rank-2, axis-0 gather path.
 *
 * The one-dimensional index selects complete rows:
 *   output[row, col] = input[index[row], col].
 *
 * TileElements is the physical capacity of one destination row. The runtime
 * valid column count is input_shape[1], while the index tile contains one
 * valid index. Keeping the source table as a two-dimensional GlobalTensor is
 * important: Coalesce::Row obtains the source-row stride from that tensor.
 */
template <typename DType, typename IType, int TileElements,
          typename IndexGlobal, typename OutputGlobal>
inline void process_gather_row(
    DType *input, IndexGlobal &index_global, OutputGlobal &output_global,
    std::uint32_t output_row, std::uint32_t input_rows,
    std::uint32_t row_width) {
  using DataTile =
      pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                pto::BLayout::RowMajor, 1, pto::DYNAMIC>;

  // Coalesce::Row accepts a row-major [1, R] index tile. This specialization
  // processes one row at a time (R == 1). Keep a 128-byte physical tile while
  // exposing only one valid index to MGATHER.
  static_assert(sizeof(IType) == sizeof(std::uint32_t),
                "Coalesce::Row requires a 32-bit index type");
  constexpr int kIndexTileElements = 128 / static_cast<int>(sizeof(IType));
  using IndexTile =
      pto::Tile<pto::Location::Vec, IType, 1, kIndexTileElements,
                pto::BLayout::RowMajor, 1, 1>;

  using TableShape =
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
  using TableStride =
      pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  using TableGlobal =
      pto::GlobalTensor<DType, TableShape, TableStride, pto::Layout::ND>;

  TableGlobal table_global(
      input, TableShape(static_cast<int>(input_rows),
                        static_cast<int>(row_width)),
      TableStride(static_cast<int>(row_width)));
  IndexGlobal tile_index_global(index_global.data() + output_row);
  OutputGlobal tile_output_global(
      output_global.data() + output_row * row_width);

  DataTile output_tile(static_cast<std::size_t>(row_width));
  IndexTile index_tile;

  TLOAD(index_tile, tile_index_global);
  MGATHER<pto::Coalesce::Row>(output_tile, table_global, index_tile);
  TSTORE(tile_output_global, output_tile);
}
#endif

/**
 * Process one physical tile. ValidElements is TileElements for a full tile and
 * kTailElements for the final partial tile. Keeping ValidElements as a template
 * parameter lets the PTO tile valid region remain a compile-time constant.
 */
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
      TMULS(index_offset, coordinate, sizeof(IType));
      MGATHER(index_tile, index_global, index_offset);
      TCVT(coordinate, index_tile);
    }
    
    // MGATHER/MSCATTER offsets use byte units.
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

} // namespace detail

/**
 * Gather a one-dimensional index along GatherDim into a contiguous output.
 *
 * The generated MGATHER offsets are expressed in bytes. The caller must ensure
 * that the input/output shapes match outside GatherDim and that their products
 * equal InputElements and OutputElements respectively.
 */
template <typename DType, typename IType, int Rank, int GatherDim, int InputElements, int OutputElements, int TileElements = 512>
void gather_v2(DType *input, IType *index, DType *output, const std::uint32_t *input_shape, const std::uint32_t *output_shape)
{
  static_assert(Rank > 0, "Rank must be positive");
  static_assert(InputElements > 0, "InputElements must be positive");
  static_assert(OutputElements > 0, "OutputElements must be positive");
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
  static_assert(GatherDim >= 0 && GatherDim < Rank,
                "GatherDim must be in [0, Rank)");

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

#if SUPERNPU_ENABLE_GATHER_V2_COALESCE_ROW
  // A rank-2 axis-0 gather with a one-dimensional index is a row selection:
  // output[r, :] = input[index[r], :]. Coalesce::Row issues one contiguous
  // row transfer instead of constructing one byte offset per output element.
  //
  // Row mode requires int32/uint32 indices. A row wider than the physical Tile
  // cannot be split portably in this mode (A5 treats validCols as row width),
  // so retain the generic element-offset implementation as the fallback.
  if constexpr (
      Rank == 2 && GatherDim == 0 &&
      (std::is_same_v<IType, std::int32_t> ||
       std::is_same_v<IType, std::uint32_t>)) {
    const std::uint32_t row_width = input_shape[1];
    if (row_width <= static_cast<std::uint32_t>(TileElements)) {
      for (std::uint32_t row = thread_id; row < output_shape[0];
           row += ThreadNum) {
        detail::process_gather_row<DType, IType, TileElements>(
            input, index_global, output_global, row, input_shape[0],
            row_width);
      }
      return;
    }
  }
#endif

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

} // namespace supernpu::tile_isa

#endif // SUPERNPU_GATHER_V2_HPP
