#ifndef JCORE_TSTORE_HPP
#define JCORE_TSTORE_HPP

#include "common/pto_tile.hpp"

using namespace pto;

#ifdef __linx
template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TSTORE_Impl(gm_shape &dst, tile_shape &src) {
  size_t rows = src.GetValidRow();
  size_t cols = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(tile_shape::isBoxedLayout == false,
                "Linx smoke TSTORE supports only unboxed tiles");

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      size_t gm_index = gm_shape::isRowMajor
                            ? row * gm_shape::RowStride + col
                            : col * gm_shape::ColStride + row;
      size_t tile_index = tile_shape::isRowMajor
                              ? row * tile_shape::RowStride + col
                              : col * tile_shape::ColStride + row;
      dst.data()[gm_index] = src.data()[tile_index];
    }
  }
}
#else
// cube left -> gm row major
template <typename gm_shape, typename tile_shape>
void __mtc__ Store2NzImpl1D(typename gm_shape::DType __out__ *dst,
                              const typename tile_shape::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  static constexpr size_t ceil_Rows =
      (tile_shape::ValidRow + tile_shape::InnerRows - 1) / tile_shape::InnerRows * tile_shape::InnerRows;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
    size_t idx_tile = sub_tile_i * ceil_Rows * inner_cols +
        sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
    size_t idx_gm = gm_shape::RowStride * j + i;
    dst[idx_gm] = tile_ptr[idx_tile];
  }
}

template <typename gm_shape, typename tile_shape>
void __mtc__ Store2ZnImpl1D(typename gm_shape::DType __out__ *dst,
                              const typename tile_shape::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  static constexpr size_t ceil_cols =
      (tile_shape::ValidCol + tile_shape::InnerCols - 1) / tile_shape::InnerCols * tile_shape::InnerCols;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
    size_t idx_gm = gm_shape::RowStride * j + i;
    size_t index = sub_tile_j * ceil_cols * inner_rows +
        sub_tile_i * tile_shape::InnerNumel + idx_i * inner_rows + idx_j;
    dst[idx_gm] = tile_ptr[index];
  }
}

//no fractal
template <typename gm_shape, typename tile_shape>
void __mtc__
TStore_Vec_ColMajor(typename gm_shape::DType __out__ *dst,
                      const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::ColStride + i;
  size_t index_tile = j * tile_shape::ColStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}

template <typename gm_shape, typename tile_shape>
void __mtc__
TStore_Vec_RowMajor(typename gm_shape::DType __out__ *dst,
                      typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::RowStride + i;
  size_t index_tile = j * tile_shape::RowStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}

// cube left -> gm row major
template <typename gm_shape, typename tile_shape>
void __mtc__ Store2NzImpl2D_Dynamic(typename gm_shape::DType __out__ *dst,
                                      const typename tile_shape::TileDType __in__ src,
                                      const size_t __in__ gm_row_stride) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
  size_t idx_tile = sub_tile_i * tile_shape::Rows * inner_cols +
      sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
  size_t idx_gm = gm_row_stride * j + i;
  dst[idx_gm] = tile_ptr[idx_tile];
}

template <typename gm_shape, typename tile_shape>
void __mtc__ Store2ZnImpl2D_Dynamic(typename gm_shape::DType __out__ *dst,
                                      const typename tile_shape::TileDType __in__ src,
                                      const size_t __in__ gm_row_stride) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
  size_t idx_gm = gm_row_stride * j + i;
  size_t index = sub_tile_j * tile_shape::Cols * inner_rows +
      sub_tile_i * tile_shape::InnerNumel + idx_i * inner_rows + idx_j;
  dst[idx_gm] = tile_ptr[index];
}

//no fractal
template <typename gm_shape, typename tile_shape>
void __mtc__ TStore_Vec_ColMajor_Dynamic(typename gm_shape::DType __out__ *dst,
                                           const typename tile_shape::TileDType __in__ src,
                                           const size_t __in__ gm_col_stride) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_col_stride + i;
  size_t index_tile = j * tile_shape::ColStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}

template <typename gm_shape, typename tile_shape>
void __mtc__ TStore_Vec_RowMajor_Dynamic(typename gm_shape::DType __out__ *dst,
                                           typename tile_shape::TileDType __in__ src,
                                           const size_t __in__ gm_row_stride) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_row_stride + i;
  size_t index_tile = j * tile_shape::RowStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}

template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TSTORE_Impl(gm_shape &dst, tile_shape &src) {
  size_t tile_rows = src.GetValidRow();
  size_t tile_cols = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
#ifdef ENABLE_TENSOR_INSTR
  static_assert(!tile_shape::isBoxedLayout || tile_shape::SFractalSize == 512,
                    "Error! Cude blk_tload:FractalSize != 512");
  static constexpr size_t tile_all_cols = tile_shape::Cols;
  static constexpr size_t tile_all_rows = tile_shape::Rows;

  if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NZ2ND,
          dst.data(),
          (dst.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8,
          src.data());
  } else if constexpr (is_Zn_layout<tile_shape>::value) { // Zn
       blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::ZN2ND,
          dst.data(),
          (dst.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8,
          src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      if constexpr (gm_shape::isRowMajor) {
        blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NORM,
          dst.data(),
          (dst.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8,
          src.data());
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (!tile_shape::isRowMajor) {
      if constexpr (!gm_shape::isRowMajor) {
        blk_tstore(tile_rows, tile_cols, tile_all_rows,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NORM,
          dst.data(),
          (dst.GetStride(4) * type_traits<typename gm_shape::DType>::bits + 7) / 8,
          src.data());
      } else {
        static_assert(!gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should colmajor");
      }
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }

#else
  if constexpr (gm_shape::RowStride == DYNAMIC || gm_shape::ColStride == DYNAMIC ||
                tile_shape::ValidRow == DYNAMIC || tile_shape::ValidCol == DYNAMIC) { // dynamic
    if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      Store2NzImpl2D_Dynamic<gm_shape, tile_shape>
          <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), dst.GetStride(3));
    } else if constexpr (is_Zn_layout<tile_shape>::value) { // Zn
      Store2ZnImpl2D_Dynamic<gm_shape, tile_shape>
          <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), dst.GetStride(3));
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        if constexpr (gm_shape::isRowMajor) {
          TStore_Vec_RowMajor_Dynamic<gm_shape, tile_shape>
              <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), dst.GetStride(3));
        } else {
          static_assert(gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should rowmajor");
        }
      } else if constexpr (!tile_shape::isRowMajor) {
        if constexpr (!gm_shape::isRowMajor) {
          TStore_Vec_ColMajor_Dynamic<gm_shape, tile_shape>
              <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data(), dst.GetStride(4));
        } else {
          static_assert(!gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should colmajor");
        }
      }
    } else {
      static_assert(tile_shape::isBoxedLayout == false,
                    "Storage layout type not supported");
    }
  } else { // static
    if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      Store2NzImpl1D<gm_shape, tile_shape>
          <<<tile_cols, 1, 1>>>(dst.data(), src.data());
    } else if constexpr (is_Zn_layout<tile_shape>::value) { // Zn
      Store2ZnImpl1D<gm_shape, tile_shape>
          <<<tile_cols, 1, 1>>>(dst.data(), src.data());
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        if constexpr (gm_shape::isRowMajor) {
          TStore_Vec_RowMajor<gm_shape, tile_shape>
              <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data());
        } else {
          static_assert(gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should rowmajor");
        }
      } else if constexpr (!tile_shape::isRowMajor) {
        if constexpr (!gm_shape::isRowMajor) {
          TStore_Vec_ColMajor<gm_shape, tile_shape>
              <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data());
        } else {
          static_assert(!gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should colmajor");
        }
      }
    } else {
      static_assert(tile_shape::isBoxedLayout == false,
                    "Storage layout type not supported");
    }
  }


#endif
}
#endif

#endif
