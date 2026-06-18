#ifndef TREM_HPP
#define TREM_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TRem_Impl(typename tile_shape::TileDType dst,
               typename tile_shape::TileDType src0,
               typename tile_shape::TileDType src1) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] =
          src0[i * tile_shape::RowStride + j] %
          src1[i * tile_shape::RowStride + j];
    }
}

template <is_tile_data_v tile_shape>
void TREM(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(std::is_same<typename tile_shape::DType, int32_t>::value ||
                    std::is_same<typename tile_shape::DType, int16_t>::value,
                    "Data type not supported");

  TRem_Impl<tile_shape>(dst.data(), src0.data(), src1.data());
}

#endif