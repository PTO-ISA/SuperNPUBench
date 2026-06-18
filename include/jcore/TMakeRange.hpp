#ifndef TMAKERANGE_HPP
#define TMAKERANGE_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__ TMakeRange_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst, 
  const typename tile_shape::DType __in__ start) {
  size_t i = blkv_get_index_x();

  blkv_get_tile_ptr(dst)[i] = start + i;
}

template <is_tile_data_v tile_shape>
void TMAKERANGE_Impl(tile_shape &dst, typename tile_shape::DType start) {
  static constexpr size_t col = tile_shape::ValidCol;

  static_assert(col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::isBoxedLayout == false,
                "TMAKERANGE not support Boxed Layout!");

  if constexpr (tile_shape::isRowMajor) {
    TMakeRange_Vec_RowMajor<tile_shape><<<col, 1, 1>>>(dst.data(), start);
  } else {
    static_assert(tile_shape::isRowMajor, "Storage layout type not supported");
  }
}

#endif

