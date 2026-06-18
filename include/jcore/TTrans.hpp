#ifndef TTRANS_HPP
#define TTRANS_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TTrans_RowMajor(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx_src = j * tile_shape_in::RowStride + i;
  size_t idx_dst = i * tile_shape_out::RowStride + j;
  blkv_get_tile_ptr(dst)[idx_dst] = blkv_get_tile_ptr(src)[idx_src];
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TTrans_ColMajor(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx_src = j * tile_shape_in::ColStride + i;
  size_t idx_dst = i * tile_shape_out::ColStride + j;
  blkv_get_tile_ptr(dst)[idx_dst] = blkv_get_tile_ptr(src)[idx_src];
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TTrans_NzLayout_Impl(typename tile_shape_out::TileDType __out__ dst,
                     const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols =
      tile_shape_in::Cols / tile_shape_in::InnerCols;
  typename tile_shape_out::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in::DType *src_tile_ptr = blkv_get_tile_ptr(src);
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t row_src =
        j * LaneNum / tile_shape_in::InnerCols + i / tile_shape_in::InnerCols;
    size_t col_src =
        k * tile_shape_in::InnerCols + i % tile_shape_in::InnerCols;

    size_t block_row_dst = col_src / tile_shape_out::InnerRows;
    size_t tail_row_dst = col_src % tile_shape_out::InnerRows;
    size_t block_col_dst = row_src / tile_shape_out::InnerCols;
    size_t tail_col_dst = row_src % tile_shape_out::InnerCols;

    size_t idx_src =
        k * tile_shape_in::Rows * tile_shape_in::InnerCols + j * LaneNum + i;
    size_t idx_dst =
        block_col_dst * tile_shape_out::Rows * tile_shape_out::InnerCols +
        block_row_dst * tile_shape_out::InnerNumel +
        tail_row_dst * tile_shape_out::InnerCols + tail_col_dst;
    dst_tile_ptr[idx_dst] = src_tile_ptr[idx_src];
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TTRANS_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(
      tile_shape_in::Rows == tile_shape_out::Cols &&
          tile_shape_in::Cols == tile_shape_out::Rows,
      "Error! Input rows != Output Columns or Input Columns != Output rows");

  static constexpr size_t row = tile_shape_in::ValidRow;
  static constexpr size_t col = tile_shape_in::ValidCol;
  static constexpr size_t row_lines =
      tile_shape_in::Rows / (LaneNum / tile_shape_in::InnerCols);

  if constexpr (is_Nz_layout<tile_shape_in>::value &&
                is_Nz_layout<tile_shape_out>::value) {
    TTrans_NzLayout_Impl<tile_shape_out, tile_shape_in>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape_in::isBoxedLayout == false &&
                       tile_shape_out::isBoxedLayout == false) {
    if constexpr (tile_shape_in::isRowMajor &&
                  tile_shape_out::isRowMajor) {
      TTrans_RowMajor<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data());
    } else {
      TTrans_ColMajor<tile_shape_out, tile_shape_in>
          <<<row, col, 1>>>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape_in::isBoxedLayout == false &&
                      tile_shape_out::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif