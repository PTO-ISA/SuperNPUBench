#ifndef TROWSUM_HPP
#define TROWSUM_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
TRowSum_NoFractal_Impl(typename tile_shape_out::TileDType __out__ dst,
                  const typename tile_shape_in::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t idx_org = i * tile_shape_in::RowStride;
  typename tile_shape_in::DType sum = blkv_get_tile_ptr(src)[idx_org];

#pragma clang loop unroll(full)
  for (size_t j = 1; j < tile_shape_in::ValidCol; ++j) {
    size_t now_idx =
        i * tile_shape_in::RowStride + j * tile_shape_in::ColStride;
    sum = sum + blkv_get_tile_ptr(src)[now_idx];
  }
  size_t idx_dst = i * tile_shape_out::RowStride;
  blkv_get_tile_ptr(dst)[idx_dst] = sum;
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TRowSum_NzLayout_Impl(typename tile_shape_out::TileDType __out__ dst,
                          const typename tile_shape_in::TileDType __in__ src) {
  unsigned i = blkv_get_index_x();
  unsigned j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape_in::Cols / tile_shape_in::InnerCols;
  typename tile_shape_out::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in::DType *src_tile_ptr = blkv_get_tile_ptr(src);
  typename tile_shape_in::DType data = src_tile_ptr[j * LaneNum + i];
#pragma clang loop unroll(full)
  for (unsigned k = 1; k < col_fract_nums; ++k) {
    size_t idx_src =
        k * tile_shape_in::Rows * tile_shape_in::InnerCols + j * LaneNum + i;
    data = data + src_tile_ptr[idx_src];
  }
//通过shuffle将第一列的所有小分型的每一行都设置为sum
#pragma clang loop unroll(full)
  for (unsigned idx = tile_shape_in::InnerCols - 1; idx >= 1; idx >>= 1) {
    data = data + blkv_shuffle_bfly(data, idx);
  }
  if (i % tile_shape_in::InnerCols == 0) {
    size_t idx_dst = j * (LaneNum / tile_shape_in::InnerCols) * tile_shape_out::RowStride +
                     i / tile_shape_in::InnerCols * tile_shape_out::RowStride;
    dst_tile_ptr[idx_dst] = data;
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(tile_shape_in::Rows == tile_shape_out::Rows,
                "Error! Input row != Output row.");
  static_assert(tile_shape_out::ValidCol == 1,
                "valid column must be 1.");
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout,
                "Not support Fractal layout");
  static constexpr size_t row = tile_shape_in::ValidRow;
  static constexpr size_t col = tile_shape_in::ValidCol;
  static constexpr size_t Y = row / (LaneNum / tile_shape_in::InnerCols);
  if constexpr (is_Nz_layout<tile_shape_in>::value) {
    TRowSum_NzLayout_Impl<tile_shape_out, tile_shape_in>
        <<<LaneNum, Y, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape_in::isBoxedLayout == false) {
    TRowSum_NoFractal_Impl<tile_shape_out, tile_shape_in>
        <<<row, 1, 1>>>(dst.data(), src.data());
  } else {
    static_assert(tile_shape_in::isBoxedLayout == false,
                  "Storage type not supported");
  }
}

#endif