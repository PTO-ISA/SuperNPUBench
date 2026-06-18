#ifndef TCVT_HPP
#define TCVT_HPP

#include "common/pto_tile.hpp"

using namespace pto;

// row major -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtRow2NzImpl1D(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape_out::InnerRows;
  static constexpr int inner_cols = tile_shape_out::InnerCols;
  static constexpr size_t ceil_rows_out =
      (tile_shape_out::ValidRow + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

    size_t idx_src = tile_shape_in::RowStride * j + i;
    size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
                     sub_tile_j * tile_shape_out::InnerNumel +
                     idx_j * inner_cols + idx_i;
    typename tile_shape_in::DType data = src_ptr[idx_src];
    dst_ptr[idx_dst] = data;
  }
}

// Nz -> row major
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2RowImpl1D(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape_in::InnerRows;
  static constexpr int inner_cols = tile_shape_in::InnerCols;
  static constexpr size_t ceil_rows_in =
      (tile_shape_in::ValidRow + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

    size_t idx_src = sub_tile_i * ceil_rows_in * inner_cols +
                     sub_tile_j * tile_shape_in::InnerNumel +
                     idx_j * inner_cols + idx_i;
    size_t idx_dst = tile_shape_out::RowStride * j + i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    dst_ptr[idx_dst] = data;
  }
}

// Nz -> Zn
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2ZnImpl1D(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  static constexpr size_t ceil_cols_out =
      (tile_shape_out::ValidCol + tile_shape_out::InnerCols - 1) / tile_shape_out::InnerCols * tile_shape_out::InnerCols;
  static constexpr size_t ceil_rows_in =
      (tile_shape_in::ValidRow + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
    size_t in_sub_tile_j = j / in_inner_rows;
    size_t in_idx_j = j % in_inner_rows;
    size_t out_sub_tile_j = j / out_inner_rows;
    size_t out_idx_j = j % out_inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                     in_sub_tile_j * tile_shape_in::InnerNumel +
                     in_idx_j * in_inner_cols + in_idx_i;
    size_t idx_dst = out_sub_tile_j * ceil_cols_out * out_inner_rows +
                     out_sub_tile_i * tile_shape_out::InnerNumel +
                     out_idx_i * out_inner_rows + out_idx_j;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    dst_ptr[idx_dst] = data;
  }
}

// Zn -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtZn2NzImpl1D(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  static constexpr size_t ceil_rows_out =
      (tile_shape_out::ValidRow + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  static constexpr size_t ceil_cols_in =
      (tile_shape_in::Cols + tile_shape_in::InnerCols - 1) / tile_shape_in::InnerCols * tile_shape_in::InnerCols;
  size_t i = blkv_get_index_x();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
    size_t in_sub_tile_j = j / in_inner_rows;
    size_t in_idx_j = j % in_inner_rows;
    size_t out_sub_tile_j = j / out_inner_rows;
    size_t out_idx_j = j % out_inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_j * ceil_cols_in * in_inner_rows +
                     in_sub_tile_i * tile_shape_in::InnerNumel +
                     in_idx_i * in_inner_rows + in_idx_j;
    size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                     out_sub_tile_j * tile_shape_out::InnerNumel +
                     out_idx_j * out_inner_cols + out_idx_i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    dst_ptr[idx_dst] = data;
  }
}

// Nz -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2NzImpl1D(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  static constexpr size_t ceil_rows_out =
      (tile_shape_out::ValidRow + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  static constexpr size_t ceil_rows_in =
      (tile_shape_in::ValidRow + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
    size_t in_sub_tile_j = j / in_inner_rows;
    size_t in_idx_j = j % in_inner_rows;
    size_t out_sub_tile_j = j / out_inner_rows;
    size_t out_idx_j = j % out_inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                     in_sub_tile_j * tile_shape_in::InnerNumel +
                     in_idx_j * in_inner_cols + in_idx_i;
    size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                     out_sub_tile_j * tile_shape_out::InnerNumel +
                     out_idx_j * out_inner_cols + out_idx_i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    dst_ptr[idx_dst] = data;
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static constexpr size_t row = tile_shape_in::ValidRow;
  static constexpr size_t col = tile_shape_in::ValidCol;

  if constexpr (tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCvtRow2NzImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                       tile_shape_out::isRowMajor) {
    TCvtNz2RowImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                       is_Zn_layout<tile_shape_out>::value) {
    TCvtNz2ZnImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Zn_layout<tile_shape_in>::value &&
                       is_Nz_layout<tile_shape_out>::value) {
    TCvtZn2NzImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                       is_Nz_layout<tile_shape_out>::value) {
    TCvtNz2NzImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else {
    static_assert((tile_shape_in::isRowMajor &&
                   is_Nz_layout<tile_shape_out>::value) ||
                      (is_Nz_layout<tile_shape_in>::value &&
                       tile_shape_out::isRowMajor) ||
                      (is_Nz_layout<tile_shape_in>::value &&
                       is_Zn_layout<tile_shape_out>::value) ||
                      (is_Zn_layout<tile_shape_in>::value &&
                       is_Nz_layout<tile_shape_out>::value) ||
                      (is_Nz_layout<tile_shape_in>::value &&
                       is_Nz_layout<tile_shape_out>::value),
                  "Storage layout type not supported");
  }
}

#endif