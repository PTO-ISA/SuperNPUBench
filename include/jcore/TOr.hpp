#ifndef TAND_HPP
#define TAND_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ TOr_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src0)[index] | blkv_get_tile_ptr(src1)[index];
}

template <typename tile_shape>
void __vec__ TOr_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src0,
  const typename tile_shape::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src0)[index] | blkv_get_tile_ptr(src1)[index];
}

template <is_tile_data_v tile_shape>
void TOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static constexpr size_t tile_rows = tile_shape::ValidRow;
  static constexpr size_t tile_cols = tile_shape::ValidCol;

  static_assert(std::is_same_v<typename tile_shape::DType, int32_t>,
                "TOR requires DType to be int32_t");
  static_assert(tile_rows != DYNAMIC && tile_cols != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::isBoxedLayout == false,
                "TOR not support Boxed Layout!");

  if constexpr (tile_shape::isRowMajor) {
    TOr_Vec_RowMajor<tile_shape><<<tile_cols, tile_rows, 1>>>
                      (dst.data(), src0.data(), src1.data());
  } else {
    TOr_Vec_ColMajor<tile_shape><<<tile_cols, tile_rows, 1>>>
                      (dst.data(), src0.data(), src1.data());
  }
}

#endif
