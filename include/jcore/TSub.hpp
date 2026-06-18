#ifndef TSUB_HPP
#define TSUB_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ Tsub_RowMajor(typename tile_shape::TileDType __out__ dst,
                           const typename tile_shape::TileDType __in__ src0,
                           const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[idx] =
      blkv_get_tile_ptr(src0)[idx] - blkv_get_tile_ptr(src1)[idx];
}

template <typename tile_shape>
void __vec__ Tsub_ColMajor(typename tile_shape::TileDType __out__ dst,
                           const typename tile_shape::TileDType __in__ src0,
                           const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[idx] =
      blkv_get_tile_ptr(src0)[idx] - blkv_get_tile_ptr(src1)[idx];
}

template <typename tile_shape>
void __vec__
Tsub_NzLayout_Impl(typename tile_shape::TileDType __out__ dst,
                   const typename tile_shape::TileDType __in__ src0,
                   const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;
  typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape::DType *src0_tile_ptr = blkv_get_tile_ptr(src0);
  typename tile_shape::DType *src1_tile_ptr = blkv_get_tile_ptr(src1);

#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t idx =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    dst_tile_ptr[idx] = src0_tile_ptr[idx] - src1_tile_ptr[idx];
  }
}

template <is_tile_data_v tile_shape>
void TSUB_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static constexpr size_t row_lines =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (is_Nz_layout<tile_shape>::value) {
    Tsub_NzLayout_Impl<tile_shape>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src0.data(), src1.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      Tsub_RowMajor<tile_shape>
          <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
    } else {
      Tsub_ColMajor<tile_shape>
          <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif