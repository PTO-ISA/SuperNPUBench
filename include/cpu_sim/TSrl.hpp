#ifndef TSRL_HPP
#define TSRL_HPP

#include <type_traits>

#include "common/pto_tile.hpp"

using namespace pto;

// Logical right shift (performed on the unsigned interpretation).
template <typename tile_shape>
void TSrl_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src, unsigned sh) {
  using U = std::make_unsigned_t<typename tile_shape::DType>;
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = static_cast<typename tile_shape::DType>(static_cast<U>(src[idx]) >> sh);
    }
}

template <typename tile_shape>
void TSrl_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src, unsigned sh) {
  using U = std::make_unsigned_t<typename tile_shape::DType>;
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = static_cast<typename tile_shape::DType>(static_cast<U>(src[idx]) >> sh);
    }
}

template <is_tile_data_v tile_shape>
void TSRL_Impl(tile_shape &dst, tile_shape &src, unsigned sh) {
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TSRL not support Boxed Layout!");

  if constexpr (tile_shape::isRowMajor) {
    TSrl_RowMajor_Imp<tile_shape>(dst.data(), src.data(), sh);
  } else {
    TSrl_ColMajor_Imp<tile_shape>(dst.data(), src.data(), sh);
  }
}

#endif
