#ifndef TSQRT_HPP
#define TSQRT_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

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
  typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape::DType *src_tile_ptr = blkv_get_tile_ptr(src);
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