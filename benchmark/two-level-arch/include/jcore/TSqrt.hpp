#ifndef TSQRT_HPP
#define TSQRT_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

#ifdef __linx
template <typename T>
T linx_tile_isqrt(T value) {
  T root = 0;
  root += (value >= static_cast<T>(1)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(4)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(9)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(16)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(25)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(36)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(49)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(64)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(81)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(100)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(121)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(144)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(169)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(196)) ? static_cast<T>(1) : static_cast<T>(0);
  root += (value >= static_cast<T>(225)) ? static_cast<T>(1) : static_cast<T>(0);
  return root;
}

template <is_tile_data_v tile_shape>
void TSQRT_Impl(tile_shape &dst, tile_shape &src) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc,
                "Unsupport ACC to be input or output here");
  static_assert(tile_shape::isBoxedLayout == false,
                "TSQRT not support Boxed Layout!");
  static_assert(std::is_integral<typename tile_shape::DType>::value,
                "Linx direct TSQRT supports integral smoke types only");

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      size_t tile_index = index<tile_shape>(i, j);
      dst.data()[tile_index] = linx_tile_isqrt(src.data()[tile_index]);
    }
  }
}
#else
template <typename tile_shape>
void __vec__ TSqrt_RowMajor(typename tile_shape::TileDType __out__ dst,
                            const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx= j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[idx] =
      blkv_fsqrt(blkv_get_tile_ptr(src)[idx]);
}

template <typename tile_shape>
void __vec__ TSqrt_ColMajor(typename tile_shape::TileDType __out__ dst,
                            const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t idx= j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[idx] =
      blkv_fsqrt(blkv_get_tile_ptr(src)[idx]);
}

template <typename tile_shape>
void __vec__
TSqrt_NzLayout_Impl(typename tile_shape::TileDType __out__ dst,
                    const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int block_cols = tile_shape::Cols / tile_shape::InnerCols;
  __vbuf__ typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape::DType *src_tile_ptr = blkv_get_tile_ptr(src);
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols; ++k) {
    size_t idx =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    dst_tile_ptr[idx] = blkv_fsqrt(src_tile_ptr[idx]);
  }
}

template <is_tile_data_v tile_shape>
void TSQRT_Impl(tile_shape &dst, tile_shape &src) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t row_lines =
      tile_shape::Rows / (LaneNum / tile_shape::InnerCols);

  if constexpr (std::is_same<typename tile_shape::DType, __half>::value ||
                std::is_same<typename tile_shape::DType, __bf16>::value ||
                std::is_same<typename tile_shape::DType, __fp32>::value) {
    if constexpr (is_Nz_layout<tile_shape>::value) {
      TSqrt_NzLayout_Impl<tile_shape>
          <<<LaneNum, row_lines, 1>>>(dst.data(), src.data());
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        TSqrt_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), src.data());
      } else {
        TSqrt_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), src.data());
      }
    } else {
      static_assert(tile_shape::isBoxedLayout == false,
                    "Storage layout type not supported");
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, __half>::value,
                  "Data type not supported");
  }
}

#endif
#endif
