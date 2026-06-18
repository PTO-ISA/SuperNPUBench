#ifndef TMINS_HPP
#define TMINS_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ TMins_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src,
  const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s_ptr = blkv_get_tile_ptr(src);
  size_t index = j * tile_shape::RowStride + i;
  d_ptr[index] = blkv_min(s_ptr[index], s);
}

template <typename tile_shape>
void __vec__ TMins_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src,
  const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s_ptr = blkv_get_tile_ptr(src);
  size_t index = j * tile_shape::ColStride + i;
  d_ptr[index] = blkv_min(s_ptr[index], s);
}

//fractal blkc impl
template <typename tile_shape>
void __vec__ TMins_NzLayout_Impl(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src,
  const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x(); 
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s_ptr = blkv_get_tile_ptr(src);

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    d_ptr[index] = blkv_min(s_ptr[index], s);
  }
}

template <is_tile_data_v tile_shape>
void TMINS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (!tile_shape::isBoxedLayout) {
    if constexpr (tile_shape::isRowMajor) {
      TMins_Vec_RowMajor<tile_shape><<<col, row, 1>>>
                        (dst.data(), src.data(), s);
    } else {
      TMins_Vec_ColMajor<tile_shape><<<row, col, 1>>>
                        (dst.data(), src.data(), s);
    } 
  } else {
    if constexpr (is_Nz_layout<tile_shape>::value) {
      TMins_NzLayout_Impl<tile_shape><<<LaneNum, Y, 1>>>
                         (dst.data(), src.data(), s);
    } else {
      static_assert(is_Nz_layout<tile_shape>::value,
                    "Layout type not supported");
    } 
  }
}

#endif