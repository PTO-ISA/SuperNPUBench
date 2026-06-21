#ifndef TPAD_HPP
#define TPAD_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
#ifndef __linx
#include <assert.h>
#endif
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in,
          typename T>
void TPAD_Impl(tile_shape_out &dst, const tile_shape_in &src, T pad_value,
               size_t up_pad, size_t left_pad, size_t down_pad,
               size_t right_pad) {
  static_assert(!tile_shape_out::isBoxedLayout &&
                    !tile_shape_in::isBoxedLayout,
                "Not support Boxed Layout!");
  static_assert(tile_shape_out::Loc == Location::Vec &&
                    tile_shape_in::Loc == Location::Vec,
                "Only VEC tile type are supported");
  static_assert(tile_shape_out::ValidRow != DYNAMIC &&
                    tile_shape_out::ValidCol != DYNAMIC &&
                    tile_shape_in::ValidRow != DYNAMIC &&
                    tile_shape_in::ValidCol != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::ValidRow >= tile_shape_in::ValidRow &&
                    tile_shape_out::ValidCol >= tile_shape_in::ValidCol,
                "Dst must cover src shape!");

  size_t src_valid_row = src.GetValidRow();
  size_t src_valid_col = src.GetValidCol();
  size_t dst_valid_row = dst.GetValidRow();
  size_t dst_valid_col = dst.GetValidCol();
  size_t after_pad_row = up_pad + src_valid_row + down_pad;
  size_t after_pad_col = left_pad + src_valid_col + right_pad;

  if (after_pad_row > dst_valid_row || after_pad_col > dst_valid_col) {
    return;
  }

  for (size_t row = 0; row < dst_valid_row; ++row) {
    for (size_t col = 0; col < dst_valid_col; ++col) {
      size_t dst_index = index<tile_shape_out>(row, col);
      bool in_src_range = row >= up_pad && row < up_pad + src_valid_row &&
                          col >= left_pad && col < left_pad + src_valid_col;
      if (in_src_range) {
        size_t src_row = row - up_pad;
        size_t src_col = col - left_pad;
        size_t src_index = index<tile_shape_in>(src_row, src_col);
        dst.data()[dst_index] = src.data()[src_index];
      } else {
        dst.data()[dst_index] =
            static_cast<typename tile_shape_out::DType>(pad_value);
      }
    }
  }
}
#else
template <typename tile_shape_out, typename tile_shape_in, typename T>
void __vec__ TPad_Vec_RowMajor(typename tile_shape_out::TileDType __out__ dst,
                            const typename tile_shape_in::TileDType __in__ src, const T __in__ pad_value,
                            const size_t __in__ up_pad, const size_t __in__ left_pad,
                            const size_t __in__ down_pad, const size_t __in__ right_pad) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = j * tile_shape_out::RowStride + i;

  size_t src_valid_row = tile_shape_in::ValidRow;
  size_t src_valid_col = tile_shape_in::ValidCol;
  size_t src_left_col_start = left_pad;
  size_t src_right_col_end = left_pad + src_valid_col;
  size_t src_up_row_start = up_pad;
  size_t src_down_row_end = up_pad + src_valid_row;
  bool in_src_range = (i >= src_left_col_start) && 
                      (i < src_right_col_end) &&
                      (j >= src_up_row_start) && 
                      (j < src_down_row_end);
  blkv_get_tile_ptr(dst)[index] = static_cast<typename tile_shape_out::DType>(pad_value);
  if (in_src_range == 1) {
    size_t src_col = i - left_pad;
    size_t src_row = j - up_pad;
    size_t src_index = src_row * tile_shape_in::RowStride + src_col;
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[src_index];
  }
}

template <typename tile_shape_out, typename tile_shape_in, typename T>
void __vec__ TPad_Vec_ColMajor(typename tile_shape_out::TileDType __out__ dst,
                              const typename tile_shape_in::TileDType __in__ src, const T __in__ pad_value,
                              const size_t __in__ up_pad, const size_t __in__ left_pad,
                              const size_t __in__ down_pad, const size_t __in__ right_pad) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = j * tile_shape_out::ColStride + i;

  size_t src_valid_row = tile_shape_in::ValidRow;
  size_t src_valid_col = tile_shape_in::ValidCol;
  size_t src_left_col_start = left_pad;
  size_t src_right_col_end = left_pad + src_valid_col;
  size_t src_up_row_start = up_pad;
  size_t src_down_row_end = up_pad + src_valid_row;
  bool in_src_range = (j >= src_left_col_start) && 
                      (j < src_right_col_end) &&
                      (i >= src_up_row_start) && 
                      (i < src_down_row_end);
  blkv_get_tile_ptr(dst)[index] = static_cast<typename tile_shape_out::DType>(pad_value);
  if (in_src_range == 1) {
    size_t src_col = j - src_left_col_start;
    size_t src_row = i - src_up_row_start;
    size_t src_index = src_col * tile_shape_in::ColStride + src_row;
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[src_index];
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in, typename T>
void TPAD_Impl(tile_shape_out &dst, const tile_shape_in &src, 
              T pad_value, size_t up_pad, size_t left_pad, 
              size_t down_pad, size_t right_pad) {
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout, 
                "Not support Boxed Layout!");
  static_assert(tile_shape_out::Loc == Location::Vec && 
                tile_shape_in::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC &&
                tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::ValidRow >= tile_shape_in::ValidRow &&
                tile_shape_out::ValidCol >= tile_shape_in::ValidCol,
                "Dst and src must be same shape!");
  
  size_t src_valid_row = src.GetValidRow();
  size_t src_valid_col = src.GetValidCol();
  size_t dst_valid_row = dst.GetValidRow();
  size_t dst_valid_col = dst.GetValidCol();
  size_t after_pad_row = up_pad + src_valid_row + down_pad;
  size_t after_pad_col = left_pad + src_valid_col + right_pad;
  assert((after_pad_row <= dst_valid_row && after_pad_col <= dst_valid_col) &&
          "Padded rows exceed destination tensor rows");

  if constexpr (tile_shape_in::isRowMajor && tile_shape_out::isRowMajor) {
    TPad_Vec_RowMajor<tile_shape_out, tile_shape_in, T>
        <<<after_pad_col, after_pad_row, 1>>>(dst.data(), src.data(), pad_value,
                                            up_pad, left_pad,
                                            down_pad, right_pad);
  } else if constexpr (!tile_shape_in::isRowMajor && !tile_shape_out::isRowMajor){
    TPad_Vec_ColMajor<tile_shape_out, tile_shape_in, T>
        <<<after_pad_row, after_pad_col, 1>>>(dst.data(), src.data(), pad_value,
                                            up_pad, left_pad,
                                            down_pad, right_pad);
  } else {
      static_assert(tile_shape_in::isRowMajor && tile_shape_out::isRowMajor,
                    "Storage layout type not supported");
  }
}
#endif
#endif
