#ifndef TAND_HPP
#define TAND_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TAnd_Impl(typename tile_shape::TileDType dst,
               typename tile_shape::TileDType src0,
               typename tile_shape::TileDType src1) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] =
          src0[i * tile_shape::RowStride + j] &
          src1[i * tile_shape::RowStride + j];
    }
}

template <is_tile_data_v tile_shape>
void TAND(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TAND not support Boxed Layout!");

  if constexpr (std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value) {
    TAnd_Impl<tile_shape>(dst.data(), src0.data(), src1.data());
  } else {
    static_assert(std::is_same<typename tile_shape::DType, int64_t>::value ||
                  std::is_same<typename tile_shape::DType, int32_t>::value ||
                  std::is_same<typename tile_shape::DType, int16_t>::value ||
                  std::is_same<typename tile_shape::DType, int8_t>::value ||
                  std::is_same<typename tile_shape::DType, unsigned long>::value ||
                  std::is_same<typename tile_shape::DType, unsigned int>::value ||
                  std::is_same<typename tile_shape::DType, unsigned short>::value ||
                  std::is_same<typename tile_shape::DType, unsigned char>::value,
                  "Only int data type are supported");
  }
}
#endif