#ifndef TABS_HPP
#define TABS_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <is_tile_data_v tile_shape>
void TABS_Impl(tile_shape &dst, tile_shape &src) {
  size_t rows = src.GetValidRow();
  size_t cols = src.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(!tile_shape::isBoxedLayout, "TABS not support Boxed Layout!");

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      size_t index = tile_shape::isRowMajor
                         ? row * tile_shape::RowStride + col
                         : col * tile_shape::ColStride + row;
      auto src_value = src.data()[index];
      auto zero = typename tile_shape::DType{};
      dst.data()[index] = src_value < zero ? -src_value : src_value;
    }
  }
}
#else
template <typename tile_shape>
void __vec__ TAbs_Vec_RowMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_fabs(blkv_get_tile_ptr(src)[index]);
}

template <typename tile_shape>
void __vec__ TAbs_Vec_ColMajor(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index] =
      blkv_fabs(blkv_get_tile_ptr(src)[index]);
}

//fractal blkc impl
template <typename tile_shape>
void __vec__ TAbs_NzLayout_Impl(
  typename tile_shape::TileDType __out__ dst,
  const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;

  __vbuf__ typename tile_shape::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *s_ptr = blkv_get_tile_ptr(src);

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    d_ptr[index] = blkv_fabs(s_ptr[index]);
  }
}

template <is_tile_data_v tile_shape> void TABS_Impl(tile_shape &dst, tile_shape &src){
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static constexpr size_t Y =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  if constexpr (!tile_shape::isBoxedLayout){
    if constexpr (std::is_same<typename tile_shape::DType, __half>::value ||
                  std::is_same<typename tile_shape::DType, __fp32>::value){
      if constexpr (tile_shape::isRowMajor)
        TAbs_Vec_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data());
      else
        TAbs_Vec_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data());
    } else {
      static_assert(std::is_same<typename tile_shape::DType, __half>::value ||
                    std::is_same<typename tile_shape::DType, __fp32>::value,
                    "Int data type not supported");
    }
  } else {
    if constexpr (std::is_same<typename tile_shape::DType, __half>::value ||
                  std::is_same<typename tile_shape::DType, __fp32>::value){
      if constexpr (is_Nz_layout<tile_shape>::value)
        TAbs_NzLayout_Impl<tile_shape><<<LaneNum, Y, 1>>>
                          (dst.data(), src.data());
      else
        static_assert(is_Nz_layout<tile_shape>::value,
                      "Layout type not supported");
    } else {
      static_assert(std::is_same<typename tile_shape::DType, __half>::value ||
                    std::is_same<typename tile_shape::DType, __fp32>::value,
                    "Int data type not supported");
    }
  }
}
#endif

#endif
