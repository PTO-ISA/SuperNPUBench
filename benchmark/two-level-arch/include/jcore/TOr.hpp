#ifndef TOR_HPP
#define TOR_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape>
void TOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  size_t rows = src0.GetValidRow();
  size_t cols = src0.GetValidCol();
  static_assert(tile_shape::Loc == Location::Vec,
                "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TOR not support Boxed Layout!");

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      size_t index = tile_shape::isRowMajor
                         ? row * tile_shape::RowStride + col
                         : col * tile_shape::ColStride + row;
      dst.data()[index] = src0.data()[index] | src1.data()[index];
    }
  }
}
#else
template <typename tile_shape>
void __vec__ TOr_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src0)[index] |
      blkv_get_tile_ptr(src1)[index];
}

template <typename tile_shape>
void __vec__ TOr_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src0)[index] |
      blkv_get_tile_ptr(src1)[index];
}

template <is_tile_data_v tile_shape>
void TOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;

  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TOR not support Boxed Layout!");

  if constexpr (std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value) {
    if constexpr (tile_shape::isRowMajor) {
      TOr_Vec_RowMajor<tile_shape><<<col, row, 1>>>
                        (dst.data(), src0.data(), src1.data());
    } else {
      TOr_Vec_ColMajor<tile_shape><<<row, col, 1>>>
                        (dst.data(), src0.data(), src1.data());
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, int64_t>::value ||
                  std::is_same<typename tile_shape::DType, int32_t>::value ||
                  std::is_same<typename tile_shape::DType, int16_t>::value ||
                  std::is_same<typename tile_shape::DType, int8_t>::value ||
                  std::is_same<typename tile_shape::DType, unsigned long>::value ||
                  std::is_same<typename tile_shape::DType, unsigned int>::value ||
                  std::is_same<typename tile_shape::DType, unsigned short>::value ||
                  std::is_same<typename tile_shape::DType, unsigned char>::value,
                  "Only int data type are supported");
  }
}
#endif

#endif
