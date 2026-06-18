#ifndef TCOPY_HPP
#define TCOPY_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __mtc__ TCopy_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
}

template <typename tile_shape>
void __mtc__ TCopy_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index];
}

//fractal blkc impl
template <typename tile_shape>
void __vec__ TCopy_NzLayout_Impl(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x(); 
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;

  typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape::DType *s_ptr = blkv_get_tile_ptr(src);

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    d_ptr[index] = s_ptr[index];
  }
}

template <is_tile_data_v tile_shape>
void TCOPY_Impl(tile_shape &dst, tile_shape &src) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (!tile_shape::isBoxedLayout) {
    if constexpr (tile_shape::isRowMajor){
      TCopy_Vec_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data());
    } else {
      TCopy_Vec_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data());
    }
  } else {
    if constexpr (is_Nz_layout<tile_shape>::value){
      TCopy_NzLayout_Impl<tile_shape><<<LaneNum, Y, 1>>>
                         (dst.data(), src.data());
    } else {
      static_assert(is_Nz_layout<tile_shape>::value,
                    "Storage layout type not supported");
    }
  }
}

#endif