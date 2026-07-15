#ifndef TDIVS_HPP
#define TDIVS_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape>
void TDIVS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(tile_shape::isBoxedLayout == false,
                "TDIVS not support Boxed Layout!");

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      size_t tile_index = index<tile_shape>(i, j);
      dst.data()[tile_index] = src.data()[tile_index] / s;
    }
  }
}
#else
template <typename tile_shape>
void __vec__ TDivsImpl_RowMajor(typename tile_shape::TileDType __out__ dst,
                                const typename tile_shape::TileDType __in__ src,
                                const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::RowStride;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index] / s;
}
template <typename tile_shape>
void __vec__ TDivsImpl_ColMajor(typename tile_shape::TileDType __out__ dst,
                                const typename tile_shape::TileDType __in__ src,
                                const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::ColStride;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index] / s;
}
template <typename tile_shape>
void __vec__ TDivs2NzImpl(typename tile_shape::TileDType __out__ dst,
                          const typename tile_shape::TileDType __in__ src,
                          const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape::Cols / tile_shape::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index] / s;
  }
}
template <is_tile_data_v tile_shape>
void TDIVS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TDivs2NzImpl<tile_shape><<<LaneNum, Y, 1>>>(dst.data(), src.data(), s);
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TDivsImpl_RowMajor<tile_shape>
          <<<col, row, 1>>>(dst.data(), src.data(), s);
    } else {
      TDivsImpl_ColMajor<tile_shape>
          <<<row, col, 1>>>(dst.data(), src.data(), s);
    }
  } else {
    static_assert(is_Nz_layout<tile_shape>::value && tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}
#endif

#endif
