#ifndef TMAX_HPP
#define TMAX_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;


template <typename tile_shape>
void __vec__
TMaxImpl_RowMajor(typename tile_shape::TileDType __out__ dst,
                  const typename tile_shape::TileDType __in__ src0,
                  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::RowStride;
  blkv_get_tile_ptr(dst)[index] =
      blkv_max(blkv_get_tile_ptr(src0)[index], blkv_get_tile_ptr(src1)[index]);
}
template <typename tile_shape>
void __vec__
TMaxImpl_ColMajor(typename tile_shape::TileDType __out__ dst,
                  const typename tile_shape::TileDType __in__ src0,
                  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::ColStride;
  blkv_get_tile_ptr(dst)[index] =
      blkv_max(blkv_get_tile_ptr(src0)[index], blkv_get_tile_ptr(src1)[index]);
}
template <typename tile_shape>
void __vec__ TMax2NzImpl(typename tile_shape::TileDType __out__ dst,
                         const typename tile_shape::TileDType __in__ src0,
                         const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape::Cols / tile_shape::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = blkv_max(blkv_get_tile_ptr(src0)[index],
                                             blkv_get_tile_ptr(src1)[index]);
  }
}
template <is_tile_data_v tile_shape>
void TMAX_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  size_t row = src0.GetValidRow();
  size_t col = src0.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TMax2NzImpl<tile_shape>
        <<<LaneNum, Y, 1>>>(dst.data(), src0.data(), src1.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TMaxImpl_RowMajor<tile_shape>
          <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
    } else {
      TMaxImpl_ColMajor<tile_shape>
          <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
    }
  } else {
    static_assert(is_Nz_layout<tile_shape>::value &&
                      tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif