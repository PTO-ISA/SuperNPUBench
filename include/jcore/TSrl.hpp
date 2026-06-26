#ifndef TSRL_HPP
#define TSRL_HPP

#include <type_traits>

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

// Logical right shift by a uniform scalar amount (per element: dst = src >> sh).
// The shift is performed on the unsigned interpretation so it is always logical.
template <typename tile_shape>
void __vec__ TSrl_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src,
  unsigned sh) {
  using U = std::make_unsigned_t<typename tile_shape::DType>;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] =
      static_cast<typename tile_shape::DType>(
          static_cast<U>(blkv_get_tile_ptr(src)[index]) >> sh);
}

template <typename tile_shape>
void __vec__ TSrl_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src,
  unsigned sh) {
  using U = std::make_unsigned_t<typename tile_shape::DType>;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] =
      static_cast<typename tile_shape::DType>(
          static_cast<U>(blkv_get_tile_ptr(src)[index]) >> sh);
}

template <is_tile_data_v tile_shape>
void TSRL_Impl(tile_shape &dst, tile_shape &src, unsigned sh) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;

  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(!tile_shape::isBoxedLayout, "TSRL not support Boxed Layout!");

  if constexpr (std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value) {
    if constexpr (tile_shape::isRowMajor) {
      TSrl_Vec_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data(), sh);
    } else {
      TSrl_Vec_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data(), sh);
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, int64_t>::value ||
                  std::is_same<typename tile_shape::DType, unsigned int>::value,
                  "Only int data type are supported");
  }
}

#endif
