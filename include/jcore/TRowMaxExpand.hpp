#ifndef TROWMAXEXPAND_HPP
#define TROWMAXEXPAND_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__
TRowMaxExpand_NoFractal_Impl(typename tile_shape::TileDType __out__ dst,
                        const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t idx_org = i * tile_shape::RowStride;
  typename tile_shape::DType max_val = blkv_get_tile_ptr(src)[idx_org];

#pragma clang loop unroll(full)
  for (size_t j = 1; j < tile_shape::ValidCol; ++j) {
    size_t idx_now = i * tile_shape::RowStride + j * tile_shape::ColStride;
    typename tile_shape::DType now_val = blkv_get_tile_ptr(src)[idx_now];
    max_val = blkv_max(max_val, now_val);
  }

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
    size_t idx = i * tile_shape::RowStride + j * tile_shape::ColStride;
    blkv_get_tile_ptr(dst)[idx] = max_val;
  }
}

template <typename tile_shape>
void __vec__
TRowMaxExpand_NzLayoput_Impl(typename tile_shape::TileDType __out__ dst,
                             const typename tile_shape::TileDType __in__ src) {
  unsigned i = blkv_get_index_x();
  unsigned j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape::Cols / tile_shape::InnerCols;
  typename tile_shape::DType *dst_tile_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape::DType *src_tile_ptr = blkv_get_tile_ptr(src);
  typename tile_shape::DType data = src_tile_ptr[j * LaneNum + i];
#pragma clang loop unroll(full)
  for (unsigned k = 1; k < col_fract_nums; ++k) {
    size_t idx_src =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    data = blkv_max(data, src_tile_ptr[idx_src]);
  }
#pragma clang loop unroll(full)
  for (unsigned idx = tile_shape::InnerCols - 1; idx >= 1; idx >>= 1) {
    data = blkv_max(data, blkv_shuffle_bfly(data, idx));
  }
#pragma clang loop unroll(full)
  for (unsigned k = 0; k < col_fract_nums; ++k) {
    size_t idx_dst =
        k * tile_shape::InnerCols * tile_shape::Rows + j * LaneNum + i;
    dst_tile_ptr[idx_dst] = data;
  }
}

// ROWMAX + EXPAND
template <is_tile_data_v tile_shape>
void TROWMAXEXPAND_Impl(tile_shape &dst, tile_shape &src) {
  static constexpr size_t row = tile_shape::ValidRow;
  static constexpr size_t col = tile_shape::ValidCol;
  static constexpr size_t Y = row / (LaneNum / tile_shape::InnerCols);
  static_assert(!tile_shape::isBoxedLayout, "Not support Fractal layout");
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TRowMaxExpand_NzLayoput_Impl<tile_shape><<<LaneNum, Y, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    TRowMaxExpand_NoFractal_Impl<tile_shape>
        <<<row, 1, 1>>>(dst.data(), src.data());
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage type not supported");
  }
}
#endif
