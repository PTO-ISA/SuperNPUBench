#ifndef TSLL_HPP
#define TSLL_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TSll_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src, unsigned sh) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = src[idx] << sh;
    }
}

template <typename tile_shape>
void TSll_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src, unsigned sh) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = src[idx] << sh;
    }
}

template <is_tile_data_v tile_shape>
void TSLL_Impl(tile_shape &dst, tile_shape &src, unsigned sh) {
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TSLL not support Boxed Layout!");

  if constexpr (tile_shape::isRowMajor) {
    TSll_RowMajor_Imp<tile_shape>(dst.data(), src.data(), sh);
  } else {
    TSll_ColMajor_Imp<tile_shape>(dst.data(), src.data(), sh);
  }
}

#endif
