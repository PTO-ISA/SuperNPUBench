#ifndef TCVT_HPP
#define TCVT_HPP

#include "common/pto_tile.hpp"
#ifndef __linx
#include "template_asm.hpp"
#endif

using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(tile_shape_in::ValidRow != DYNAMIC &&
                    tile_shape_in::ValidCol != DYNAMIC &&
                    tile_shape_out::ValidRow != DYNAMIC &&
                    tile_shape_out::ValidCol != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_in::Loc != Location::Acc,
                "Linx direct TCVT smoke does not support ACC input");
  static_assert(tile_shape_out::Loc != Location::Acc,
                "ACC can not be output tile!");
  static_assert(tile_shape_in::ValidRow == tile_shape_out::ValidRow &&
                    tile_shape_in::ValidCol == tile_shape_out::ValidCol,
                "TCVT direct path requires matching logical shapes");

  for (size_t row = 0; row < tile_shape_in::ValidRow; ++row) {
    for (size_t col = 0; col < tile_shape_in::ValidCol; ++col) {
      size_t src_index = index<tile_shape_in>(row, col);
      size_t dst_index = index<tile_shape_out>(row, col);
      dst.data()[dst_index] =
          static_cast<typename tile_shape_out::DType>(src.data()[src_index]);
    }
  }
}
#else
template <typename, typename = void>
struct blkc_has_data_member : std::false_type {};

template <typename T>
struct blkc_has_data_member<T, std::void_t<decltype(std::declval<T>().data)>>
    : std::true_type {};

template <typename T>
inline constexpr bool blkc_has_data_member_v = blkc_has_data_member<T>::value;


template <typename T, typename IndexT>
inline void blkc_assign_elem(__vbuf__ T* ptr, IndexT idx, T value) {
  if constexpr (blkc_has_data_member<T>::value) {
    ptr[idx].data = value.data;
  } else{
    ptr[idx] = value;
  }
}

template <typename T, typename IndexT, typename U>
inline void blkc_assign_cast(__vbuf__ T* ptr, IndexT idx, U value) {
  T tmp = static_cast<T>(value);
  blkc_assign_elem(ptr, idx, tmp);
}

#define BLKC_ASSIGN_CAST(tile, idx, value) \
  blkc_assign_cast(blkv_get_tile_ptr(tile), idx, value)

// 直接 ptr 版本：用于替换 dst_ptr[idx_dst] = data;
#define BLKC_ASSIGN_PTR(ptr, idx, value) \
  blkc_assign_elem((ptr), (idx), (value))

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
    __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

    size_t idx_src = tile_shape_in::RowStride * j + i;
    size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
                     sub_tile_j * tile_shape_out::InnerNumel +
                     idx_j * inner_cols + idx_i;
    typename tile_shape_in::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_CAST_COMMON(dst_ptr, idx_dst, data);
  }
}

// col major -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtCol2NzImpl1D(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape_out::InnerRows;
  static constexpr int inner_cols = tile_shape_out::InnerCols;
  static constexpr size_t ceil_rows_out =
      (tile_shape_out::ValidRow + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  size_t j = blkv_get_index_x();

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

#pragma clang loop unroll(full)
  for (size_t i = 0; i < tile_shape_out::ValidCol; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;

    size_t idx_src = tile_shape_in::ColStride * i + j;
    size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
                     sub_tile_j * tile_shape_out::InnerNumel +
                     idx_j * inner_cols + idx_i;
    typename tile_shape_in::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
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
    __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

    size_t idx_src = sub_tile_i * ceil_rows_in * inner_cols +
                     sub_tile_j * tile_shape_in::InnerNumel +
                     idx_j * inner_cols + idx_i;
    size_t idx_dst = tile_shape_out::RowStride * j + i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
  }
}

// Nz -> col major
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2ColImpl1D(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape_in::InnerRows;
  static constexpr int inner_cols = tile_shape_in::InnerCols;
  static constexpr size_t ceil_rows_in =
      (tile_shape_in::ValidRow + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t j = blkv_get_index_x();

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  
#pragma clang loop unroll(full)
  for (size_t i = 0; i < tile_shape_in::ValidCol; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;
    size_t idx_src = sub_tile_i * ceil_rows_in * inner_cols +
                     sub_tile_j * tile_shape_in::InnerNumel +
                     idx_j * inner_cols + idx_i;
    size_t idx_dst = tile_shape_out::ColStride * i + j;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
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
    __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                     in_sub_tile_j * tile_shape_in::InnerNumel +
                     in_idx_j * in_inner_cols + in_idx_i;
    size_t idx_dst = out_sub_tile_j * ceil_cols_out * out_inner_rows +
                     out_sub_tile_i * tile_shape_out::InnerNumel +
                     out_idx_i * out_inner_rows + out_idx_j;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
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
    __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_j * ceil_cols_in * in_inner_rows +
                     in_sub_tile_i * tile_shape_in::InnerNumel +
                     in_idx_i * in_inner_rows + in_idx_j;
    size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                     out_sub_tile_j * tile_shape_out::InnerNumel +
                     out_idx_j * out_inner_cols + out_idx_i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
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
    __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
    __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
    size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                     in_sub_tile_j * tile_shape_in::InnerNumel +
                     in_idx_j * in_inner_cols + in_idx_i;
    size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                     out_sub_tile_j * tile_shape_out::InnerNumel +
                     out_idx_j * out_inner_cols + out_idx_i;
    typename tile_shape_out::DType data = src_ptr[idx_src];
    BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
  }
}


//TCVT for Dynamic
// row major -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtRow2NzImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src, const size_t dst_valid_row) {
  static constexpr int inner_rows = tile_shape_out::InnerRows;
  static constexpr int inner_cols = tile_shape_out::InnerCols;
  const size_t ceil_rows_out =
      (dst_valid_row + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

  size_t idx_src = tile_shape_in::RowStride * j + i;
  size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
                    sub_tile_j * tile_shape_out::InnerNumel +
                    idx_j * inner_cols + idx_i;
  typename tile_shape_in::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// col major -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtCol2NzImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src, const size_t dst_valid_row) {
  static constexpr int inner_rows = tile_shape_out::InnerRows;
  static constexpr int inner_cols = tile_shape_out::InnerCols;
  const size_t ceil_rows_out =
      (dst_valid_row + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  size_t j = blkv_get_index_x();
  size_t i = blkv_get_index_y();

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t idx_src = tile_shape_in::ColStride * i + j;
  size_t idx_dst = sub_tile_i * ceil_rows_out * inner_cols +
                    sub_tile_j * tile_shape_out::InnerNumel +
                    idx_j * inner_cols + idx_i;
  typename tile_shape_in::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// Nz -> row major
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2RowImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src, const size_t src_valid_row) {
  static constexpr int inner_rows = tile_shape_in::InnerRows;
  static constexpr int inner_cols = tile_shape_in::InnerCols;
  const size_t ceil_rows_in =
      (src_valid_row + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);

  size_t idx_src = sub_tile_i * ceil_rows_in * inner_cols +
                    sub_tile_j * tile_shape_in::InnerNumel +
                    idx_j * inner_cols + idx_i;
  size_t idx_dst = tile_shape_out::RowStride * j + i;
  typename tile_shape_out::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// Nz -> col major
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2ColImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src, const size_t src_valid_row) {
  static constexpr int inner_rows = tile_shape_in::InnerRows;
  static constexpr int inner_cols = tile_shape_in::InnerCols;
  const size_t ceil_rows_in =
      (src_valid_row + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t j = blkv_get_index_x();
  size_t i = blkv_get_index_y();

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  
  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;
  size_t idx_src = sub_tile_i * ceil_rows_in * inner_cols +
                    sub_tile_j * tile_shape_in::InnerNumel +
                    idx_j * inner_cols + idx_i;
  size_t idx_dst = tile_shape_out::ColStride * i + j;
  typename tile_shape_out::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// Nz -> Zn
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2ZnImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src, const size_t src_valid_row, const size_t dst_valid_col) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  const size_t ceil_cols_out =
      (dst_valid_col + tile_shape_out::InnerCols - 1) / tile_shape_out::InnerCols * tile_shape_out::InnerCols;
  const size_t ceil_rows_in =
      (src_valid_row + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

  size_t in_sub_tile_j = j / in_inner_rows;
  size_t in_idx_j = j % in_inner_rows;
  size_t out_sub_tile_j = j / out_inner_rows;
  size_t out_idx_j = j % out_inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                    in_sub_tile_j * tile_shape_in::InnerNumel +
                    in_idx_j * in_inner_cols + in_idx_i;
  size_t idx_dst = out_sub_tile_j * ceil_cols_out * out_inner_rows +
                    out_sub_tile_i * tile_shape_out::InnerNumel +
                    out_idx_i * out_inner_rows + out_idx_j;
  typename tile_shape_out::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// Zn -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtZn2NzImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src, const size_t dst_valid_row, const size_t src_valid_col) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  const size_t ceil_rows_out =
      (dst_valid_row + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  const size_t ceil_cols_in =
      (src_valid_col + tile_shape_in::InnerCols - 1) / tile_shape_in::InnerCols * tile_shape_in::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

  size_t in_sub_tile_j = j / in_inner_rows;
  size_t in_idx_j = j % in_inner_rows;
  size_t out_sub_tile_j = j / out_inner_rows;
  size_t out_idx_j = j % out_inner_rows;

  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  size_t idx_src = in_sub_tile_j * ceil_cols_in * in_inner_rows +
                    in_sub_tile_i * tile_shape_in::InnerNumel +
                    in_idx_i * in_inner_rows + in_idx_j;
  size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                    out_sub_tile_j * tile_shape_out::InnerNumel +
                    out_idx_j * out_inner_cols + out_idx_i;
  typename tile_shape_out::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

// Nz -> Nz
template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TCvtNz2NzImpl2D_Dynamic(typename tile_shape_out::TileDType __out__ dst,
                const typename tile_shape_in::TileDType __in__ src, const size_t dst_valid_row, const size_t src_valid_row) {
  static constexpr int in_inner_rows = tile_shape_in::InnerRows;
  static constexpr int in_inner_cols = tile_shape_in::InnerCols;
  static constexpr int out_inner_rows = tile_shape_out::InnerRows;
  static constexpr int out_inner_cols = tile_shape_out::InnerCols;
  const size_t ceil_rows_out =
      (dst_valid_row + tile_shape_out::InnerRows - 1) / tile_shape_out::InnerRows * tile_shape_out::InnerRows;
  const size_t ceil_rows_in =
      (src_valid_row + tile_shape_in::InnerRows - 1) / tile_shape_in::InnerRows * tile_shape_in::InnerRows;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t in_sub_tile_i = i / in_inner_cols;
  size_t in_idx_i = i % in_inner_cols;
  size_t out_sub_tile_i = i / out_inner_cols;
  size_t out_idx_i = i % out_inner_cols;

  size_t in_sub_tile_j = j / in_inner_rows;
  size_t in_idx_j = j % in_inner_rows;
  size_t out_sub_tile_j = j / out_inner_rows;
  size_t out_idx_j = j % out_inner_rows;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape_in::DType *src_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  size_t idx_src = in_sub_tile_i * ceil_rows_in * in_inner_cols +
                    in_sub_tile_j * tile_shape_in::InnerNumel +
                    in_idx_j * in_inner_cols + in_idx_i;
  size_t idx_dst = out_sub_tile_i * ceil_rows_out * out_inner_cols +
                    out_sub_tile_j * tile_shape_out::InnerNumel +
                    out_idx_j * out_inner_cols + out_idx_i;
  typename tile_shape_out::DType data = src_ptr[idx_src];
  BLKC_ASSIGN_PTR(dst_ptr, idx_dst, data);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(tile_shape_out::Loc != Location::Acc, "ACC can not be output tile!");
  size_t row = src.GetValidRow();
  size_t col = src.GetValidCol();
  if constexpr (tile_shape_in::Loc == Location::Acc) {
    static_assert(is_Nz_layout<tile_shape_in>::value, "ACC layout must be Nz!");
    if constexpr (!tile_shape_out::isBoxedLayout && tile_shape_out::isRowMajor) {
      blk_acccvt(row, col,
          type_traits<typename tile_shape_in::DType>::TypeCode,
          type_traits<typename tile_shape_out::DType>::TypeCode,
          LayoutCvtEnum::NZ2ND,
          true,
          dst.data(),
          src.data());
    } else if constexpr (!tile_shape_out::isBoxedLayout && !tile_shape_out::isRowMajor) {
      blk_acccvt(row, col,
          type_traits<typename tile_shape_in::DType>::TypeCode,
          type_traits<typename tile_shape_out::DType>::TypeCode,
          LayoutCvtEnum::NZ2DN,
          true,
          dst.data(),
          src.data());
    } else if constexpr (is_Zn_layout<tile_shape_out>::value) {
      blk_acccvt(row, col,
          type_traits<typename tile_shape_in::DType>::TypeCode,
          type_traits<typename tile_shape_out::DType>::TypeCode,
          LayoutCvtEnum::NZ2ZN,
          true,
          dst.data(),
          src.data());
    } else if constexpr (is_Nz_layout<tile_shape_out>::value) {
      blk_acccvt(row, col,
          type_traits<typename tile_shape_in::DType>::TypeCode,
          type_traits<typename tile_shape_out::DType>::TypeCode,
          LayoutCvtEnum::NORM,
          true,
          dst.data(),
          src.data());
    } else {
      static_assert("TCVT ACC output Storage layout type not supported");
    }
    return;
  }

#ifdef ENABLE_HARDEN
  if constexpr (tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCVT_ND2NZ(dst, src);
  } else if constexpr (!tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCVT_DN2NZ(dst, src);   
  } else if constexpr (tile_shape_in::isRowMajor &&
                is_Zn_layout<tile_shape_out>::value) {
    TCVT_ND2ZN(dst, src);
  } else if constexpr (!tile_shape_in::isRowMajor &&
                is_Zn_layout<tile_shape_out>::value) {
    TCVT_DN2ZN(dst, src);
  } else if constexpr (!tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCVT_DN2NZ(dst, src);
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                      tile_shape_out::isRowMajor) {
    TCVT_NZ2ND(dst, src);
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                      !tile_shape_out::isRowMajor) {
    TCVT_NZ2DN(dst, src);
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                      is_Zn_layout<tile_shape_out>::value) {
    TCVT_NZ2ZN(dst, src);
  } else if constexpr (is_Zn_layout<tile_shape_in>::value &&
                      is_Nz_layout<tile_shape_out>::value) {
    TCVT_ZN2NZ(dst, src);
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                      is_Nz_layout<tile_shape_out>::value) {
    TCVT_NZ2NZ(dst, src);                
  } else {
    static_assert(tile_shape_in::Loc != Location::Acc, "Unreachable code!");
    static_assert((tile_shape_in::isRowMajor &&
                  is_Nz_layout<tile_shape_out>::value) ||
                  (!tile_shape_in::isRowMajor &&
                  is_Nz_layout<tile_shape_out>::value) ||
                  (tile_shape_in::isRowMajor &&
                  is_Zn_layout<tile_shape_out>::value) ||
                  (!tile_shape_in::isRowMajor &&
                  is_Zn_layout<tile_shape_out>::value) ||
                  (!tile_shape_in::isRowMajor &&
                  is_Nz_layout<tile_shape_out>::value) ||
                  (is_Nz_layout<tile_shape_in>::value &&
                  tile_shape_out::isRowMajor)          ||
                  (is_Nz_layout<tile_shape_in>::value &&
                  !tile_shape_out::isRowMajor)         ||
                  (is_Nz_layout<tile_shape_in>::value &&
                  is_Zn_layout<tile_shape_out>::value) ||
                  (is_Zn_layout<tile_shape_in>::value &&
                  is_Nz_layout<tile_shape_out>::value) ||
                  (is_Nz_layout<tile_shape_in>::value &&
                  is_Nz_layout<tile_shape_out>::value),
                  "Storage layout type not supported");
  }
#endif

  if constexpr (tile_shape_in::ValidRow == DYNAMIC || tile_shape_in::ValidCol == DYNAMIC ||
                tile_shape_out::ValidRow == DYNAMIC || tile_shape_out::ValidCol == DYNAMIC) {
    if constexpr (tile_shape_in::isRowMajor &&
                  is_Nz_layout<tile_shape_out>::value) {
      TCvtRow2NzImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data(), dst.GetValidRow());
    } else if constexpr (!tile_shape_in::isRowMajor &&
                  is_Nz_layout<tile_shape_out>::value) {
      TCvtCol2NzImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<row, col, 1>>>(dst.data(), src.data(), dst.GetValidRow());               
    } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                        tile_shape_out::isRowMajor) {
      TCvtNz2RowImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data(), src.GetValidRow());
    } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                        !tile_shape_out::isRowMajor) {
      TCvtNz2ColImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<row, col, 1>>>(dst.data(), src.data(), src.GetValidRow());
    } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                        is_Zn_layout<tile_shape_out>::value) {
      TCvtNz2ZnImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data(),src.GetValidRow(), dst.GetValidCol());
    } else if constexpr (is_Zn_layout<tile_shape_in>::value &&
                        is_Nz_layout<tile_shape_out>::value) {
      TCvtZn2NzImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data(),dst.GetValidRow(), src.GetValidCol());
    } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                        is_Nz_layout<tile_shape_out>::value) {
      TCvtNz2NzImpl2D_Dynamic<tile_shape_out, tile_shape_in>
          <<<col, row, 1>>>(dst.data(), src.data(),dst.GetValidRow(), src.GetValidRow());
    } else {
      static_assert(tile_shape_in::Loc != Location::Acc, "Unreachable code!");
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
    return;
  }
  
  if constexpr (tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCvtRow2NzImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (!tile_shape_in::isRowMajor &&
                is_Nz_layout<tile_shape_out>::value) {
    TCvtCol2NzImpl1D<tile_shape_out, tile_shape_in>
        <<<row, 1, 1>>>(dst.data(), src.data());               
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                       tile_shape_out::isRowMajor) {
    TCvtNz2RowImpl1D<tile_shape_out, tile_shape_in>
        <<<col, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Nz_layout<tile_shape_in>::value &&
                       !tile_shape_out::isRowMajor) {
    TCvtNz2ColImpl1D<tile_shape_out, tile_shape_in>
        <<<row, 1, 1>>>(dst.data(), src.data());
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
    static_assert(tile_shape_in::Loc != Location::Acc, "Unreachable code!");
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
#endif
