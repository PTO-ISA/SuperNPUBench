#ifndef TCOPYOUT_HPP
#define TCOPYOUT_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename gm_shape, typename tile_shape>
void CopyOut2NzImpl1D(typename gm_shape::DType *dst,
                      const typename tile_shape::TileDType src) {
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
      const typename tile_shape::DType *tile_ptr = src;
      size_t idx_gm = gm_shape::kRowStride * j + i;
      size_t idx_tile = sub_tile_i * tile_shape::Rows * inner_cols +
          sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
      dst[idx_gm] = tile_ptr[idx_tile];
    }
  }
}

template <typename gm_shape, typename tile_shape>
void TCopyOut_ColMajor_Impl(typename gm_shape::DType *dst,
                            typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx_gm = i * gm_shape::kColStride + j;
      size_t idx_tile = i * tile_shape::ColStride + j;
      dst[idx_gm] = src[idx_tile];
    }
}

template <typename gm_shape, typename tile_shape>
void TCopyOut_RowMajor_Impl(typename gm_shape::DType *dst,
                            typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx_gm = i * gm_shape::kRowStride + j;
      size_t idx_tile = i * tile_shape::RowStride + j;
      dst[idx_gm] = src[idx_tile];
    }
}

template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TCOPYOUT_Impl(gm_shape &dst, tile_shape &src) {
  if constexpr (is_Nz_layout<tile_shape>::value) {
    if constexpr (gm_shape::isRowMajor) {
      CopyOut2NzImpl1D<gm_shape, tile_shape>(dst.data(), src.data());
    } else {
      static_assert(gm_shape::isRowMajor,
                    "Storage layout type not supported, gm should rowmajor");
    }
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (gm_shape::isRowMajor) {
      TCopyOut_RowMajor_Impl<gm_shape, tile_shape>(dst.data(), src.data());
    } else {
      TCopyOut_ColMajor_Impl<gm_shape, tile_shape>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Data type not supported");
  }
}

#endif