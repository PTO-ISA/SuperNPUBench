#ifndef TCI_HPP
#define TCI_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape, int desc>
void TCI_Impl(typename tile_shape::TileDType dst,
            const typename tile_shape::DType s) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
        size_t index = i * tile_shape::RowStride + j;
        if constexpr (desc) {
            // 降序：s - index
            dst[index] = s - static_cast<typename tile_shape::DType>(index);
        } else {
            // 升序：s + index
            dst[index] = s + static_cast<typename tile_shape::DType>(index);
        }
    }
}

template <is_tile_data_v tile_shape, typename T, int descending>
void TCI(tile_shape &dst, T s) {
  TCI_Impl<tile_shape, descending>(dst.data(), s);
}

#endif