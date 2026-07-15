#ifndef TMIN_HPP
#define TMIN_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
#include <cstdint>

using namespace pto;

template <typename tile_shape>
void __vec__ TMin_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s0_ptr = blkv_get_tile_ptr(src0);
  __vbuf__ typename tile_shape::DType *s1_ptr = blkv_get_tile_ptr(src1);
  size_t index = j * tile_shape::RowStride + i;
  d_ptr[index] = blkv_min(s0_ptr[index], s1_ptr[index]);
}

template <typename tile_shape>
void __vec__ TMin_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s0_ptr = blkv_get_tile_ptr(src0);
  __vbuf__ typename tile_shape::DType *s1_ptr = blkv_get_tile_ptr(src1);
  size_t index = j * tile_shape::ColStride + i;
  d_ptr[index] = blkv_min(s0_ptr[index], s1_ptr[index]);
}

//fractal blkc impl
template <typename tile_shape>
void __vec__ TMin_NzLayout_Impl(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s0_ptr = blkv_get_tile_ptr(src0);
  __vbuf__ typename tile_shape::DType *s1_ptr = blkv_get_tile_ptr(src1);

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    d_ptr[index] = blkv_min(s0_ptr[index], s1_ptr[index]);
  }
}

template <is_tile_data_v tile_shape>
void TMIN_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (!tile_shape::isBoxedLayout) {
    if constexpr (tile_shape::isRowMajor) {
      TMin_Vec_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
      } else {
      TMin_Vec_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
    }
  } else {
    if constexpr (is_Nz_layout<tile_shape>::value) {
      TMin_NzLayout_Impl<tile_shape><<<LaneNum, Y, 1>>>
                                    (dst.data(), src0.data(), src1.data());
    } else {
      static_assert(is_Nz_layout<tile_shape>::value,
                    "Layout type not supported");
    }
  }
}

#endif