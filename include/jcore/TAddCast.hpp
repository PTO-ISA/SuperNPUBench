#ifndef TADDCAST_HPP
#define TADDCAST_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in0, typename tile_shape_in1>
void __vec__ TAddCast_Vec_RowMajor(
  typename tile_shape_out::TileDType __out__ dst,
  const typename tile_shape_in0::TileDType __in__ src0,
  const typename tile_shape_in1::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::RowStride + i;
  blkv_get_tile_ptr(dst)[index] =
      static_cast<typename tile_shape_out::DType>(blkv_get_tile_ptr(src0)[index]) +
      static_cast<typename tile_shape_out::DType>(blkv_get_tile_ptr(src1)[index]);
}

template <typename tile_shape_out, typename tile_shape_in0, typename tile_shape_in1>
void __vec__ TAddCast_Vec_ColMajor(
  typename tile_shape_out::TileDType __out__ dst,
  const typename tile_shape_in0::TileDType __in__ src0,
  const typename tile_shape_in1::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape_out::ColStride + i;
  blkv_get_tile_ptr(dst)[index] =
      static_cast<typename tile_shape_out::DType>(blkv_get_tile_ptr(src0)[index]) +
      static_cast<typename tile_shape_out::DType>(blkv_get_tile_ptr(src1)[index]);
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in0,
          is_tile_data_v tile_shape_in1>
void TADDCAST_Impl(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1) {
  static constexpr uint16_t tile_rows = tile_shape_out::ValidRow;
  static constexpr uint16_t tile_cols = tile_shape_out::ValidCol;

  static_assert(tile_rows != DYNAMIC && tile_cols != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_out::isBoxedLayout == false,
                "TADDCAST not support Boxed Layout!");

  if constexpr (tile_shape_out::isRowMajor) {
    TAddCast_Vec_RowMajor<tile_shape_out, tile_shape_in0, tile_shape_in1><<<tile_cols, tile_rows, 1>>>
                      (dst.data(), src0.data(), src1.data());
  } else {
    TAddCast_Vec_ColMajor<tile_shape_out, tile_shape_in0, tile_shape_in1><<<tile_cols, tile_rows, 1>>>
                      (dst.data(), src0.data(), src1.data());
  }
}

#endif
