#ifndef TADD_HPP
#define TADD_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ TAdd_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src0)[index] +
      blkv_get_tile_ptr(src1)[index];
}

template <typename tile_shape>
void __vec__ TAdd_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_get_tile_ptr(src0)[index] +
      blkv_get_tile_ptr(src1)[index];
}

//fractal blkc impl
template <typename tile_shape>
void __vec__ TAdd_NzLayout_Impl(
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
    d_ptr[index] = s0_ptr[index] + s1_ptr[index];
  }
}

template <is_tile_data_v tile_shape>
void TADD_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  size_t tile_rows = src0.GetValidRow();
  size_t tile_cols = src0.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static_assert(tile_shape::isBoxedLayout == false,
                "TADD not support Boxed Layout!");

  if constexpr (tile_shape::isRowMajor) {
    TAdd_Vec_RowMajor<tile_shape><<<tile_cols, tile_rows, 1>>>
                      (dst.data(), src0.data(), src1.data());
  } else {
    TAdd_Vec_ColMajor<tile_shape><<<tile_rows, tile_cols, 1>>>
                      (dst.data(), src0.data(), src1.data());
  }

}

#endif
