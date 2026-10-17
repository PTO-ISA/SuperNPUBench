#ifndef SUPERNPU_COPY_TILE_HPP
#define SUPERNPU_COPY_TILE_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace supernpu::tile_isa::detail {

// Keep physical capacity fixed. validCols restricts both the load and store,
// including a short segment that must not touch the next row or a guard.
template <typename DType, int TileElements>
inline void copy_contiguous_tile(DType *input, DType *output,
                                 std::uint32_t validCols) {
  using Tile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                         pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
  using Global = pto::global_tensor<DType, pto::RowMajor<1, TileElements>>;
  Global src(input), dst(output);
  Tile data(validCols);
  TLOAD(data, src);
  TSTORE(dst, data);
}

template <typename DType, int Rows, int Cols>
inline void copy_strided_rows(DType *input, DType *output,
                              std::uint32_t validRows,
                              std::uint32_t inputStride,
                              std::uint32_t outputStride) {
  using Tile = pto::Tile<pto::Location::Vec, DType, Rows, Cols,
                         pto::BLayout::RowMajor, pto::DYNAMIC, Cols>;
  // Static shape avoids the current API's unsupported default construction
  // of dynamic Shape; only the GM row strides are dynamic.
  using Shape = pto::Shape<1, 1, 1, Rows, Cols>;
  using Stride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  using Global = pto::GlobalTensor<DType, Shape, Stride, pto::Layout::ND>;
  Global src(input, Shape{}, Stride(inputStride));
  Global dst(output, Shape{}, Stride(outputStride));
  Tile data(validRows);
  TLOAD(data, src);
  TSTORE(dst, data);
}

} // namespace supernpu::tile_isa::detail
#endif // SUPERNPU_COPY_TILE_HPP
