#ifndef TEXPANDROW_HPP
#define TEXPANDROW_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandRow_RowImpl(typename tile_shape_out::TileDType __out__ dst,
                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape_out::RowStride;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src)[i * tile_shape_in::ColStride];
}
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandRow_ColImpl(typename tile_shape_out::TileDType __out__ dst,
                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape_out::ColStride;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src)[j * tile_shape_in::ColStride];
}
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandRow2Nzimpl(typename tile_shape_out::TileDType __out__ dst,
                  const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape_out::Cols / tile_shape_out::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index = k * tile_shape_out::Rows * tile_shape_out::InnerCols +
                   j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(
        src)[(i % tile_shape_out::InnerCols + k * tile_shape_out::InnerCols) *
             tile_shape_in::ColStride];
  }
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDROW_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert((tile_shape_out::Cols == tile_shape_in::Cols) &&
                    (tile_shape_out::ValidCol == tile_shape_in::ValidCol),
                "Error! Cude A:Columns != Cude B:Columns");
  static_assert(tile_shape_out::ValidCol == tile_shape_in::ValidCol,
                "Error! Cude A:ValidCol != Cude B:ValidCol");
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout,
                "Not support Fractal layout");
  static_assert(tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
                tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::Loc != Location::Acc && tile_shape_in::Loc != Location::Acc, 
              "Unsupport ACC to be input or output here");
  static constexpr size_t row = tile_shape_out::ValidRow;
  static constexpr size_t col = tile_shape_out::ValidCol;
  static constexpr size_t Y =
      tile_shape_out::Rows / (LaneNum / tile_shape_out::InnerCols);
  if constexpr (is_Nz_layout<tile_shape_out>::value) {
    TExpandRow2Nzimpl<tile_shape_out, tile_shape_in>
        <<<64, Y, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape_out::isBoxedLayout == false) {
    if constexpr (tile_shape_out::isRowMajor) {
      TExpandRow_RowImpl<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data());
    } else {
      TExpandRow_ColImpl<tile_shape_out, tile_shape_in>
          <<<row, col, 1>>>(dst.data(), src.data());
    }
  } else {
    static_assert(is_Nz_layout<tile_shape_out>::value &&
                      tile_shape_out::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif