#ifndef TCI_HPP
#define TCI_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape, typename T, int descending>
void TCI_Impl(tile_shape &dst, T s) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;

  static_assert(std::is_same<typename tile_shape::DType, T>::value,
                "Dst and scalar must be same data type!");
  static_assert((descending == 0) || (descending == 1),
                "descending must be 0 or 1!");
  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc == Location::Vec,
                "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TCI not support Boxed Layout!");
  static_assert(std::is_same<typename tile_shape::DType, int32_t>::value ||
                    std::is_same<typename tile_shape::DType, uint32_t>::value ||
                    std::is_same<typename tile_shape::DType, int16_t>::value ||
                    std::is_same<typename tile_shape::DType, uint16_t>::value,
                "Data type not supported");

  for (size_t row_idx = 0; row_idx < row; ++row_idx) {
    for (size_t col_idx = 0; col_idx < col; ++col_idx) {
      size_t tile_index = index<tile_shape>(row_idx, col_idx);
      if constexpr (descending) {
        dst.data()[tile_index] =
            s - static_cast<typename tile_shape::DType>(tile_index);
      } else {
        dst.data()[tile_index] =
            s + static_cast<typename tile_shape::DType>(tile_index);
      }
    }
  }
}
#else
template <typename tile_shape, int desc>
void __vec__ TCIImpl_RowMajor(typename tile_shape::TileDType __out__ dst,
                                const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;

  if constexpr (desc) {
      // 降序：s - index
      blkv_get_tile_ptr(dst)[index] = s - static_cast<typename tile_shape::DType>(index);
  } else {
      // 升序：s + index
      blkv_get_tile_ptr(dst)[index] = s + static_cast<typename tile_shape::DType>(index);
  }
}
template <typename tile_shape, int desc>
void __vec__ TCIImpl_ColMajor(typename tile_shape::TileDType __out__ dst,
                                const typename tile_shape::DType __in__ s) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;

  if constexpr (desc) {
      // 降序：s - index
      blkv_get_tile_ptr(dst)[index] = s - static_cast<typename tile_shape::DType>(index);
  } else {
      // 升序：s + index
      blkv_get_tile_ptr(dst)[index] = s + static_cast<typename tile_shape::DType>(index);
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
      TCIImpl_RowMajor<tile_shape, descending>
          <<<col, 1, 1>>>(dst.data(), s);
    } else {
      TCIImpl_ColMajor<tile_shape, descending>
          <<<row, 1, 1>>>(dst.data(), s);
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
#endif
