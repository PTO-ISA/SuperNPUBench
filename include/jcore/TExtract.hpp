#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TExtract_RowMajor_Imp(
    typename tile_shape_out::TileDType __out__ dst,
    const typename tile_shape_in::TileDType __in__ src,
    const size_t __in__ offset_row, const size_t __in__ offset_col) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx_src =
      (offset_row + j) * tile_shape_in::RowStride + offset_col + i;
  size_t idx_dst = j * tile_shape_out::RowStride + i;
  blkv_get_tile_ptr(dst)[idx_dst] = blkv_get_tile_ptr(src)[idx_src];
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TExtract_ColMajor_Imp(
    typename tile_shape_out::TileDType __out__ dst,
    const typename tile_shape_in::TileDType __in__ src,
    const size_t __in__ offset_row, const size_t __in__ offset_col) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx_src =
      (offset_col + j) * tile_shape_in::ColStride + offset_row + i;
  size_t idx_dst = j * tile_shape_out::ColStride + i;
  blkv_get_tile_ptr(dst)[idx_dst] = blkv_get_tile_ptr(src)[idx_src];
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TExtract_NzLayout_Imp(
    typename tile_shape_out::TileDType __out__ dst,
    const typename tile_shape_in::TileDType __in__ src,
    const size_t __in__ offset_row, const size_t __in__ offset_col) {

  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t start_subtile_i = offset_row / tile_shape_in::InnerRows;
  size_t start_idx_i = offset_row % tile_shape_in::InnerRows;
  size_t start_subtile_j = offset_col / tile_shape_in::InnerCols;
  size_t start_idx_j = offset_col % tile_shape_in::InnerCols;
  size_t offset =
      start_subtile_j * tile_shape_in::Rows * tile_shape_in::InnerCols +
      start_subtile_i * tile_shape_in::InnerNumel +
      start_idx_i * tile_shape_in::InnerCols + start_idx_j;
  static constexpr int block_cols =
      tile_shape_out::Cols / tile_shape_out::InnerCols;

  typename tile_shape_out::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in::DType *src_tile_ptr = blkv_get_tile_ptr(src);
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t idx_src = k * tile_shape_in::Rows * tile_shape_in::InnerCols +
                     j * LaneNum + i + offset +
                     k * start_subtile_i * tile_shape_in::InnerNumel;
    size_t idx_dst = k * tile_shape_out::Rows * tile_shape_out::InnerCols +
                     j * LaneNum + i;
    dst_tile_ptr[idx_dst] = src_tile_ptr[idx_src];
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXTRACT_Impl(tile_shape_out &dst, tile_shape_in &src, size_t offset_row,
              size_t offset_col) {
  static constexpr size_t dst_row = tile_shape_out::ValidRow;
  static constexpr size_t dst_col = tile_shape_out::ValidCol;
  static constexpr size_t row_lines =
      tile_shape_out::Rows / (LaneNum / tile_shape_out::InnerCols);
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout,
                "Not support Fractal layout");
  if constexpr (is_Nz_layout<tile_shape_in>::value &&
                is_Nz_layout<tile_shape_in>::value) {
    TExtract_NzLayout_Imp<tile_shape_out, tile_shape_in>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src.data(), offset_row,
                                    offset_col);
  } else if constexpr (tile_shape_in ::isBoxedLayout == false &&
                       tile_shape_out ::isBoxedLayout == false) {
    if constexpr (tile_shape_in::isRowMajor &&
                  tile_shape_out::isRowMajor) {
      TExtract_RowMajor_Imp<tile_shape_out, tile_shape_in>
          <<<dst_col, dst_row, 1>>>(dst.data(), src.data(), offset_row,
                                    offset_col);
    } else if constexpr (!tile_shape_in::isRowMajor &&
                         !tile_shape_out::isRowMajor) {
      TExtract_ColMajor_Imp<tile_shape_out, tile_shape_in>
          <<<dst_row, dst_col, 1>>>(dst.data(), src.data(), offset_row,
                                    offset_col);
    } else {
      static_assert(tile_shape_in::isRowMajor &&
                        tile_shape_out::isRowMajor,
                    "Storage layout type not supported");
    }
  } else {
    static_assert(tile_shape_in::isBoxedLayout == false &&
                      tile_shape_out ::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif