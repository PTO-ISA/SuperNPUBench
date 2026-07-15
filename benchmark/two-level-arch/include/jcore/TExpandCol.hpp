#ifndef TEXPANDCOL_HPP
#define TEXPANDCOL_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDCOL_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert((tile_shape_out::Rows == tile_shape_in::Rows) &&
                    (tile_shape_out::ValidRow == tile_shape_in::ValidRow),
                "Error! Cude A:Rows != Cude B:Rows");
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout,
                "Not support Fractal layout");
  static_assert(tile_shape_out::Loc != Location::Acc &&
                    tile_shape_in::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");

  size_t row = dst.GetValidRow();
  size_t col = dst.GetValidCol();
  for (size_t row_idx = 0; row_idx < row; ++row_idx) {
    for (size_t col_idx = 0; col_idx < col; ++col_idx) {
      size_t dst_index = index<tile_shape_out>(row_idx, col_idx);
      size_t src_index = index<tile_shape_in>(row_idx, 0);
      dst.data()[dst_index] = src.data()[src_index];
    }
  }
}
#else
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandCol_RowImpl(typename tile_shape_out::TileDType __out__ dst,
                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape_out::RowStride;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[j*tile_shape_in::RowStride];
}
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandCol_ColImpl(typename tile_shape_out::TileDType __out__ dst,
                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape_out::ColStride;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[i*tile_shape_in::RowStride];
}
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TExpandCol2Nzimpl(typename tile_shape_out::TileDType __out__ dst,
                  const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape_out::Cols / tile_shape_out::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index = k * tile_shape_out::Rows * tile_shape_out::InnerCols +
                   j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] =
        blkv_get_tile_ptr(src)[(i / tile_shape_out::InnerCols +
                                j * (LaneNum / tile_shape_out::InnerCols)) *
                               tile_shape_in::RowStride];
  }
}
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDCOL_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert((tile_shape_out::Rows == tile_shape_in::Rows) &&
                    (tile_shape_out::ValidRow == tile_shape_in::ValidRow),
                "Error! Cude A:Rows != Cude B:Rows");
  static_assert(tile_shape_out::ValidRow == tile_shape_in::ValidRow,
                "Error! Cude A:ValidRow != Cude B:ValidRow");
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout,
                "Not support Fractal layout");
  static_assert(tile_shape_out::Loc != Location::Acc && tile_shape_in::Loc != Location::Acc,
              "Unsupport ACC to be input or output here");
  size_t row = dst.GetValidRow();
  size_t col = dst.GetValidCol();
  static constexpr size_t Y =
      tile_shape_out::Rows / (LaneNum / tile_shape_out::InnerCols);
  if constexpr (is_Nz_layout<tile_shape_out>::value) {
    TExpandCol2Nzimpl<tile_shape_out, tile_shape_in>
        <<<64, Y, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape_out::isBoxedLayout == false) {
    if constexpr (tile_shape_out::isRowMajor) {
      TExpandCol_RowImpl<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data());
    } else {
      TExpandCol_ColImpl<tile_shape_out, tile_shape_in>
          <<<row, col, 1>>>(dst.data(), src.data());
    }
  } else {
    static_assert(is_Nz_layout<tile_shape_out>::value &&
                      tile_shape_out::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif
#endif
