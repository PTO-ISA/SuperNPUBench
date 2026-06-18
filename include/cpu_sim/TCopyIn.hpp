#ifndef TCOPYIN_HPP
#define TCOPYIN_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape, typename gm_shape>
void CopyInRow2NzImpl1D(typename tile_shape::TileDType dst,
                     const typename gm_shape::DType *src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  for (size_t i = 0; i < tile_shape::ValidCol; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t sub_tile_j = j / inner_rows;
      size_t idx_j = j % inner_rows;
      // sub_tile_j and idx_j can be replaced with a constant after loop
      // unrolling
      typename tile_shape::DType *tile_ptr = dst;
      size_t idx_gm = gm_shape::kRowStride * j + i;
      size_t idx_tile = sub_tile_i * tile_shape::Rows * inner_cols +
          sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
      tile_ptr[idx_tile] = src[idx_gm];
    }
  }
}

template <typename tile_shape, typename gm_shape>
void CopyInRow2ZnImpl1D(typename tile_shape::TileDType dst,
                     const typename gm_shape::DType *src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    size_t sub_tile_i = i / inner_rows;
    size_t idx_i = i % inner_rows;
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t sub_tile_j = j / inner_cols;
      size_t idx_j = j % inner_cols;
      // sub_tile_j and idx_j can be replaced with a constant after loop
      // unrolling
      typename tile_shape::DType *tile_ptr = dst;
      size_t idx_gm = gm_shape::kRowStride * i + j;
      size_t idx_tile = sub_tile_i * tile_shape::Cols * inner_rows +
          sub_tile_j * tile_shape::InnerNumel + idx_j * inner_rows + idx_i;
      tile_ptr[idx_tile] = src[idx_gm];
    }
  }
}

template <typename tile_shape, typename gm_shape>
void CopyInCol2ZnImpl1D(typename tile_shape::TileDType dst,
                     const typename gm_shape::DType *src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    size_t sub_tile_i = i / inner_rows;
    size_t idx_i = i % inner_rows;
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t sub_tile_j = j / inner_cols;
      size_t idx_j = j % inner_cols;
      // sub_tile_j and idx_j can be replaced with a constant after loop
      // unrolling
      typename tile_shape::DType *tile_ptr = dst;
      size_t idx_gm = gm_shape::kColStride * j + i;
      size_t idx_tile = sub_tile_i * tile_shape::Cols * inner_rows +
          sub_tile_j * tile_shape::InnerNumel + idx_j * inner_rows + idx_i;
      tile_ptr[idx_tile] = src[idx_gm];
    }
  }
}

template <typename tile_shape, typename gm_shape>
void TCopyIn_RowMajor_Impl(typename tile_shape::TileDType dst,
                           const typename gm_shape::DType *src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx_gm = i * gm_shape::kRowStride + j;
      size_t idx_tile = i * tile_shape::RowStride + j;
      dst[idx_tile] = src[idx_gm];
    }
}

template <typename tile_shape, typename gm_shape>
void TCopyIn_ColMajor_Impl(typename tile_shape::TileDType dst,
                           const typename gm_shape::DType *src) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx_gm = i * gm_shape::kColStride + j;
      size_t idx_tile = i * tile_shape::ColStride + j;
      dst[idx_tile] = src[idx_gm];
    }
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TCOPYIN_Impl(tile_shape &dst, const gm_shape &src) {
  if constexpr (is_Nz_layout<tile_shape>::value) {
    if constexpr (gm_shape::isRowMajor) {
      CopyInRow2NzImpl1D<tile_shape, gm_shape>(dst.data(), src.data());
    } else {
      static_assert(gm_shape::isRowMajor,
                    "Storage layout type not supported, gm should rowmajor");
    }
  } else if constexpr (is_Zn_layout<tile_shape>::value) {
    if constexpr (!gm_shape::isRowMajor) {
      CopyInCol2ZnImpl1D<tile_shape, gm_shape>(dst.data(), src.data());
    } else {
      CopyInRow2ZnImpl1D<tile_shape, gm_shape>(dst.data(), src.data());
    }
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (gm_shape::isRowMajor) {
      TCopyIn_RowMajor_Impl<tile_shape, gm_shape>(dst.data(), src.data());
    } else {
      TCopyIn_ColMajor_Impl<tile_shape, gm_shape>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Data type not supported");
  }
}

#endif