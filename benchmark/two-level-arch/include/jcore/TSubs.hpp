#ifndef TSUBS_HPP
#define TSUBS_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape>
void TSUBS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  size_t rows = src.GetValidRow();
  size_t cols = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(!tile_shape::isBoxedLayout, "TSUBS not support Boxed Layout!");

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      size_t index = tile_shape::isRowMajor
                         ? row * tile_shape::RowStride + col
                         : col * tile_shape::ColStride + row;
      dst.data()[index] = src.data()[index] - s;
    }
  }
}
#else
template <typename tile_shape>
void __vec__ TSubs_RowMajor(typename tile_shape::TileDType __out__ dst,
                            const typename tile_shape::TileDType __in__ src,
                            const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[idx] = blkv_get_tile_ptr(src)[idx] - s;
}

template <typename tile_shape>
void __vec__ TSubs_ColMajor(typename tile_shape::TileDType __out__ dst,
                            const typename tile_shape::TileDType __in__ src,
                            const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[idx] = blkv_get_tile_ptr(src)[idx] - s;
}

template <typename tile_shape>
void __vec__
TSubs_NzLayout_Impl(typename tile_shape::TileDType __out__ dst,
                    const typename tile_shape::TileDType __in__ src,
                    const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;
  __vbuf__ typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *src_tile_ptr = blkv_get_tile_ptr(src);
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t idx =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    dst_tile_ptr[idx] = src_tile_ptr[idx] - s;
  }
}

template <is_tile_data_v tile_shape>
void TSUBS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t row_lines =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (is_Nz_layout<tile_shape>::value) {
    TSubs_NzLayout_Impl<tile_shape>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src.data(), s);
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TSubs_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data(), s);
    } else {
      TSubs_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data(), s);
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}
#endif

#endif
