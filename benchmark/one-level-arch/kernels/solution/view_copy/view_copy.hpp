/**
 * @file view_copy.hpp
 * @brief Copy between strided tensor views with Tile ISA gather/scatter.
 */

#ifndef SUPERNPU_VIEW_COPY_HPP
#define SUPERNPU_VIEW_COPY_HPP

#include "solution/common/copy_tile.hpp"

#include <cstdint>

namespace supernpu::tile_isa {
namespace detail {

// Compute only the outer coordinates, once per rectangular transfer.
inline void view_outer_offsets(std::uint32_t outer, int lastDim,
                                const std::uint32_t *shape,
                                const std::uint32_t *inputStride,
                                const std::uint32_t *outputStride,
                                std::uint32_t &src, std::uint32_t &dst) {
  src = dst = 0;
  for (int dim = lastDim; dim >= 0; --dim) {
    const std::uint32_t coordinate = outer % shape[dim];
    outer /= shape[dim];
    src += coordinate * inputStride[dim];
    dst += coordinate * outputStride[dim];
  }
}

template <typename DType, int Elements, int TileElements, int Cols>
void copy_view_rows(DType *input, DType *output, int rowDim,
                    const std::uint32_t *shape,
                    const std::uint32_t *inputStride,
                    const std::uint32_t *outputStride,
                    std::uint32_t tid) {
  constexpr std::uint32_t rowsPerTile = TileElements / Cols;
  const std::uint32_t rows = shape[rowDim];
  const std::uint32_t blocks = (rows + rowsPerTile - 1) / rowsPerTile;
  const std::uint32_t groups = Elements / (rows * Cols);
  for (std::uint32_t job = tid; job < groups * blocks; job += 4) {
    const std::uint32_t row = (job % blocks) * rowsPerTile;
    std::uint32_t src, dst;
    view_outer_offsets(job / blocks, rowDim - 1, shape, inputStride,
                       outputStride, src, dst);
    const std::uint32_t valid = rows - row < rowsPerTile
                                  ? rows - row : rowsPerTile;
    copy_strided_rows<DType, rowsPerTile, Cols>(
        input + src + row * inputStride[rowDim],
        output + dst + row * outputStride[rowDim], valid,
        inputStride[rowDim], outputStride[rowDim]);
  }
}

/**
 * Process one physical tile. ValidElements is TileElements for a full tile and
 * kTailElements for the final partial tile. Keeping ValidElements as a template
 * parameter lets the PTO tile valid region remain a compile-time constant.
 */
template <typename DType, int Rank, int TileElements, int ValidElements,
          bool OutputContiguous,
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
  if constexpr (!OutputContiguous)
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

    // Public strides are elements; PTO 0.58.4+ indexed TLSU uses bytes.
    TMULS(input_contribution, coordinate,
          input_global_stride[dim] * static_cast<std::uint32_t>(sizeof(DType)));
    if constexpr (!OutputContiguous)
      TMULS(output_contribution, coordinate,
            output_global_stride[dim] * static_cast<std::uint32_t>(sizeof(DType)));

    TADD(input_offset_tile, input_offset_tile, input_contribution);
    if constexpr (!OutputContiguous)
      TADD(output_offset_tile, output_offset_tile, output_contribution);

    logical_stride *= shape[dim];
  }

  MGATHER(output_tile, input_global, input_offset_tile);
  if constexpr (OutputContiguous) {
    OutputGlobal tile_output(output_global.data() + linear_base);
    TSTORE(tile_output, output_tile);
  } else {
    MSCATTER(output_global, output_tile, output_offset_tile);
  }
}

} // namespace detail

/**
 * Copy Elements logical elements between two strided views of the same shape.
 *
 * input_offset/output_offset and internal gather/scatter offsets are bytes;
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

  using InputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
  using OutputGlobal =
      pto::global_tensor<DType, pto::RowMajor<1, Elements>>;

  InputGlobal input_global(input + input_offset / sizeof(DType));
  OutputGlobal output_global(output + output_offset / sizeof(DType));

  constexpr std::uint32_t ThreadNum = 4;
  std::uint32_t thread_id = get_thread_idx();

  // Collapse the suffix that is contiguous in BOTH views. Size-one axes do
  // not constrain a view's stride. A segment is never allowed to cross an
  // outer stride discontinuity.
  std::uint32_t contiguous = 1;
  int rowDim = Rank - 1;
  for (; rowDim >= 0; --rowDim) {
    if (shape[rowDim] == 1) continue;
    if (input_global_stride[rowDim] != contiguous ||
        output_global_stride[rowDim] != contiguous) break;
    contiguous *= shape[rowDim];
  }
  DType *srcBase = input + input_offset / sizeof(DType);
  DType *dstBase = output + output_offset / sizeof(DType);
  const bool batchRows = rowDim >= 0 && contiguous < TileElements &&
      ((contiguous == 64 && TileElements % 64 == 0) ||
       (contiguous == 128 && TileElements % 128 == 0) ||
       (contiguous == 256 && TileElements % 256 == 0));
  if ((!batchRows && contiguous * sizeof(DType) >= 128) || rowDim < 0) {
    const std::uint32_t chunks = (contiguous + TileElements - 1) / TileElements;
    const std::uint32_t segments = Elements / contiguous;
    for (std::uint32_t job = thread_id; job < segments * chunks; job += ThreadNum) {
      const std::uint32_t col = (job % chunks) * TileElements;
      std::uint32_t src, dst;
      detail::view_outer_offsets(job / chunks, rowDim, shape,
                                input_global_stride, output_global_stride, src, dst);
      const std::uint32_t valid = contiguous - col < TileElements
                                    ? contiguous - col : TileElements;
      detail::copy_contiguous_tile<DType, TileElements>(
          srcBase + src + col, dstBase + dst + col, valid);
    }
    return;
  }

  // Batch narrow contiguous rows into one Tile instead of constructing an
  // element-wise gather/scatter index. These widths cover common head sizes.
  if (rowDim >= 0) {
    if constexpr (TileElements >= 64 && TileElements % 64 == 0) {
      if (contiguous == 64) {
        detail::copy_view_rows<DType, Elements, TileElements, 64>(
            srcBase, dstBase, rowDim, shape, input_global_stride,
            output_global_stride, thread_id);
        return;
      }
    }
    if constexpr (TileElements >= 128 && TileElements % 128 == 0) {
      if (contiguous == 128) {
        detail::copy_view_rows<DType, Elements, TileElements, 128>(
            srcBase, dstBase, rowDim, shape, input_global_stride,
            output_global_stride, thread_id);
        return;
      }
    }
    if constexpr (TileElements >= 256 && TileElements % 256 == 0) {
      if (contiguous == 256) {
        detail::copy_view_rows<DType, Elements, TileElements, 256>(
            srcBase, dstBase, rowDim, shape, input_global_stride,
            output_global_stride, thread_id);
        return;
      }
    }
  }

  bool outputContiguous = true;
  std::uint32_t stride = 1;
  for (int dim = Rank - 1; dim >= 0; --dim) {
    if (shape[dim] != 1 && output_global_stride[dim] != stride)
      outputContiguous = false;
    stride *= shape[dim];
  }

  for (int tile = static_cast<int>(thread_id); tile < kFullTiles;
       tile += ThreadNum) {
    if (outputContiguous)
      detail::process_view_copy_tile<DType, Rank, TileElements, TileElements, true>(
          input_global, output_global, tile * TileElements, shape,
          input_global_stride, output_global_stride);
    else
      detail::process_view_copy_tile<DType, Rank, TileElements, TileElements, false>(
          input_global, output_global, tile * TileElements, shape,
          input_global_stride, output_global_stride);
  }

  if constexpr (kTailElements != 0) {
    constexpr int kTailOwner = kFullTiles % ThreadNum;
    if (thread_id == static_cast<std::uint32_t>(kTailOwner)) {
      if (outputContiguous)
        detail::process_view_copy_tile<DType, Rank, TileElements, kTailElements, true>(
            input_global, output_global, kFullTiles * TileElements, shape,
            input_global_stride, output_global_stride);
      else
        detail::process_view_copy_tile<DType, Rank, TileElements, kTailElements, false>(
            input_global, output_global, kFullTiles * TileElements, shape,
            input_global_stride, output_global_stride);
    }
  }
}

} // namespace supernpu::tile_isa

#endif // SUPERNPU_VIEW_COPY_HPP
