#ifndef TEXP_HPP
#define TEXP_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <typename T>
T linx_tile_iexp(T value) {
  T result = static_cast<T>(1);
  result +=
      (value >= static_cast<T>(1)) ? static_cast<T>(2) : static_cast<T>(0);
  result +=
      (value >= static_cast<T>(2)) ? static_cast<T>(4) : static_cast<T>(0);
  result +=
      (value >= static_cast<T>(3)) ? static_cast<T>(13) : static_cast<T>(0);
  result +=
      (value >= static_cast<T>(4)) ? static_cast<T>(35) : static_cast<T>(0);
  result +=
      (value >= static_cast<T>(5)) ? static_cast<T>(93) : static_cast<T>(0);
  return result;
}

template <is_tile_data_v tile_shape>
void TEXP_Impl(tile_shape &dst, tile_shape &src) {
  size_t rows = src.GetValidRow();
  size_t cols = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(!tile_shape::isBoxedLayout, "TEXP not support Boxed Layout!");
  static_assert(std::is_integral<typename tile_shape::DType>::value,
                "Linx direct TEXP supports integral smoke types only");

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      size_t tile_index = tile_shape::isRowMajor
                              ? row * tile_shape::RowStride + col
                              : col * tile_shape::ColStride + row;
      dst.data()[tile_index] = linx_tile_iexp(src.data()[tile_index]);
    }
  }
}
#else
template <typename tile_shape>
void __vec__
TExpImpl_RowMajor(typename tile_shape::TileDType __out__ dst,
                  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::RowStride;
  blkv_get_tile_ptr(dst)[index] = blkv_fexp(blkv_get_tile_ptr(src)[index]);
}
template <typename tile_shape>
void __vec__
TExpImpl_ColMajor(typename tile_shape::TileDType __out__ dst,
                  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::ColStride;
  blkv_get_tile_ptr(dst)[index] = blkv_fexp(blkv_get_tile_ptr(src)[index]);
}
template <typename tile_shape>
void __vec__ TExp2NzImpl(typename tile_shape::TileDType __out__ dst,
                         const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape::Cols / tile_shape::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = blkv_fexp(blkv_get_tile_ptr(src)[index]);
  }
}
template <is_tile_data_v tile_shape> void TEXP_Impl(tile_shape &dst, tile_shape &src) {
  size_t row = src.GetValidRow();
  size_t col = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);
  if constexpr (std::is_same<typename tile_shape::DType, __half>::value ||
                std::is_same<typename tile_shape::DType, __bf16>::value ||
                std::is_same<typename tile_shape::DType, __fp32>::value) {
    if constexpr (is_Nz_layout<tile_shape>::value) {
      TExp2NzImpl<tile_shape><<<LaneNum, Y, 1>>>(dst.data(), src.data());
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        TExpImpl_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data());
      } else {
        TExpImpl_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data());
      }
    } else {
      static_assert(is_Nz_layout<tile_shape>::value &&
                        tile_shape::isBoxedLayout == false,
                    "Storage layout type not supported");
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, __half>::value,
                  "Data type not supported");
  }
}

#endif

#endif
