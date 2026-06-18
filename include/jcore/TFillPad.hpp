#ifndef TFILLPAD_HPP
#define TFILLPAD_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TFillPad_Vec_RowMajor(typename tile_shape_out::TileDType __out__ dst,
                                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::RowStride + i;

  if (i < tile_shape_in::ValidCol && j < tile_shape_in::ValidRow) {
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
  } else {
    blkv_get_tile_ptr(dst)[index] = 0;
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TFillPad_Vec_ColMajor(typename tile_shape_out::TileDType __out__ dst,
                                   const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::ColStride + i;

  if (i < tile_shape_in::ValidRow && j < tile_shape_in::ValidCol) {
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
  } else {
    blkv_get_tile_ptr(dst)[index] = 0;
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TFillPad_Vec_RowMajor_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                                           const typename tile_shape_in::TileDType __in__ src,
                                           const size_t __in__ valid_col,
                                           const size_t __in__ valid_row) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::RowStride + i;

  if (i < valid_col && j < valid_row) {
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
  } else {
    blkv_get_tile_ptr(dst)[index] = 0;
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TFillPad_Vec_ColMajor_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                                           const typename tile_shape_in::TileDType __in__ src,
                                           const size_t __in__ valid_row,
                                           const size_t __in__ valid_col) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::ColStride + i;

  if (i < valid_row && j < valid_col) {
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
  } else {
    blkv_get_tile_ptr(dst)[index] = 0;
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TFILLPAD_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout, "Not support Boxed Layout!");
  static_assert(tile_shape_out::Rows == tile_shape_in::Rows && tile_shape_out::Cols == tile_shape_in::Cols,
                "Dst and src must be same shape!");
  static_assert(tile_shape_out::PadVal == PadValue::Zero, "Only support pad zero!");
  static_assert(tile_shape_out::Loc != Location::Acc && tile_shape_in::Loc != Location::Acc, 
              "Unsupport ACC to be input or output here");
  static constexpr size_t dst_row = tile_shape_out::Rows;
  static constexpr size_t dst_col = tile_shape_out::Cols;
  
  size_t src_row = src.GetValidRow();
  size_t src_col = src.GetValidCol();

  if constexpr (tile_shape_in::ValidRow == DYNAMIC || tile_shape_in::ValidCol == DYNAMIC) { // dynamic
    if constexpr (tile_shape_out::isRowMajor) {
      static_assert(tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_RowMajor_Dynamic<tile_shape_out, tile_shape_in>
          <<<dst_col, dst_row, 1>>>(dst.data(), src.data(), src_col, src_row);
    } else {
      static_assert(!tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_ColMajor_Dynamic<tile_shape_out, tile_shape_in>
          <<<dst_row, dst_col, 1>>>(dst.data(), src.data(), src_row, src_col);
    }
  } else { // static
    if constexpr (tile_shape_out::isRowMajor) {
      static_assert(tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_RowMajor<tile_shape_out, tile_shape_in><<<dst_col, dst_row, 1>>>(dst.data(), src.data());
    } else {
      static_assert(!tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_ColMajor<tile_shape_out, tile_shape_in><<<dst_row, dst_col, 1>>>(dst.data(), src.data());
    }
  }
  
}

#endif