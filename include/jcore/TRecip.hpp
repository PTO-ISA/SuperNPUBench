#ifndef TRECIP_HPP
#define TRECIP_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ TRecip_RowMajor(typename tile_shape::TileDType __out__ dst,
                             const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[idx] = 1.0 / blkv_get_tile_ptr(src)[idx];
}

template <typename tile_shape>
void __vec__ TRecip_ColMajor(typename tile_shape::TileDType __out__ dst,
                             const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[idx] = 1.0 / blkv_get_tile_ptr(src)[idx];
}

template <typename tile_shape>
void __vec__
TRecip_NzLayout_Impl(typename tile_shape::TileDType __out__ dst,
                     const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;
  __vbuf__ typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *src_tile_ptr = blkv_get_tile_ptr(src);
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t idx =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    dst_tile_ptr[idx] = 1.0 / src_tile_ptr[idx];
  }
}

template <is_tile_data_v tile_shape>
void TRECIP_Impl(tile_shape &dst, tile_shape &src) {
  size_t row = src.GetValidRow();
  size_t col = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  size_t row_lines =
      row / (LaneNum / tile_shape::InnerCols);
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TRecip_NzLayout_Impl<tile_shape>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TRecip_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data());
    } else {
      TRecip_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif