#ifndef TADD_HPP
#define TADD_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TAdd_Impl(typename tile_shape::TileDType dst,
               typename tile_shape::TileDType src0,
               typename tile_shape::TileDType src1) {
  for (uint16_t i = 0; i < tile_shape::Rows; ++i)
    for (uint16_t j = 0; j < tile_shape::Cols; ++j) {
      dst[i * tile_shape::RowStride + j] =
          src0[i * tile_shape::RowStride + j] +
          src1[i * tile_shape::RowStride + j];
    }
}

template <is_tile_data_v tile_shape>
void TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  TAdd_Impl<tile_shape>(dst.data(), src0.data(), src1.data());
}

#endif