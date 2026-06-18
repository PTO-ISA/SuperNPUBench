#ifndef TCI_HPP
#define TCI_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape, int desc>
void TCI_RowMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      if constexpr (desc) {
        dst[idx] = s - static_cast<typename tile_shape::DType>(idx);
      } else {
        dst[idx] = s + static_cast<typename tile_shape::DType>(idx);
      }
    }
}
template <typename tile_shape, int desc>
void TCI_ColMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      if constexpr (desc) {
        dst[idx] = s - static_cast<typename tile_shape::DType>(idx);
      } else {
        dst[idx] = s + static_cast<typename tile_shape::DType>(idx);
      }
    }
}

template <is_tile_data_v tile_shape, typename T, int descending>
void TCI_Impl(tile_shape &dst, T s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;

  static_assert(std::is_same<typename tile_shape::DType, T>::value, "Dst and scalar must be same data type!");
  static_assert((descending == 0) || (descending == 1), "descending must be 0 or 1!");
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TCI not support Boxed Layout!");
if constexpr (std::is_same<typename tile_shape::DType, int32_t>::value ||
              std::is_same<typename tile_shape::DType, uint32_t>::value ||
              std::is_same<typename tile_shape::DType, int16_t>::value ||
              std::is_same<typename tile_shape::DType, uint16_t>::value) {
    if constexpr (tile_shape::isRowMajor) {
      TCI_RowMajor_Imp<tile_shape, descending>(dst.data(), s);
    } else {
      TCI_ColMajor_Imp<tile_shape, descending>(dst.data(), s);
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, int32_t>::value ||
                  std::is_same<typename tile_shape::DType, uint32_t>::value ||
                  std::is_same<typename tile_shape::DType, int16_t>::value ||
                  std::is_same<typename tile_shape::DType, uint16_t>::value,
                  "Data type not supported");
  }
}

#endif