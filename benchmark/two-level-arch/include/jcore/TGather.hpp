#ifndef TGATHER_HPP
#define TGATHER_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_dst, typename tile_shape_src,
          typename tile_shape_indices>
void __vec__ TGather_Vec_RowMajor(
    typename tile_shape_dst::TileDType __out__ dst,
    const typename tile_shape_src::TileDType __in__ src,
    const typename tile_shape_indices::TileDType __in__ indices) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape_dst::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape_src::DType *s0_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_indices::DType *si_ptr = blkv_get_tile_ptr(indices);
  size_t index = j * tile_shape_indices::RowStride + i;
  size_t idx = si_ptr[index] * tile_shape_src::RowStride + i;
  d_ptr[index] = s0_ptr[idx];
}

template <typename tile_shape_dst, typename tile_shape_src,
          typename tile_shape_indices>
void __vec__ TGather_Vec_ColMajor(
    typename tile_shape_dst::TileDType __out__ dst,
    const typename tile_shape_src::TileDType __in__ src,
    const typename tile_shape_indices::TileDType __in__ indices) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  __vbuf__ typename tile_shape_dst::DType *d_ptr = blkv_get_tile_ptr(dst);
  __vbuf__ typename tile_shape_src::DType *s0_ptr = blkv_get_tile_ptr(src);
  __vbuf__ typename tile_shape_indices::DType *si_ptr = blkv_get_tile_ptr(indices);
  size_t index = j * tile_shape_indices::ColStride + i;
  size_t idx = si_ptr[index] + j * tile_shape_src::ColStride;
  d_ptr[index] = s0_ptr[idx];
}

template <is_tile_data_v tile_shape_dst, is_tile_data_v tile_shape_src,
          is_tile_data_v tile_shape_indices>
void TGATHER_Impl(tile_shape_dst &dst, tile_shape_src &src,
                  tile_shape_indices &indices) {
  static_assert(tile_shape_dst::Rows == tile_shape_indices::Rows &&
                    tile_shape_dst::Cols == tile_shape_indices::Cols,
                "Error! Output shape != Indices(indices) shape");
  static_assert(!tile_shape_dst::isBoxedLayout && !tile_shape_src::isBoxedLayout &&
                   !tile_shape_indices::isBoxedLayout,
                "Not support Fractal layout");
  static_assert(tile_shape_dst::ValidRow != DYNAMIC && tile_shape_dst::ValidCol != DYNAMIC &&
                tile_shape_src::ValidRow != DYNAMIC && tile_shape_src::ValidCol != DYNAMIC &&
                tile_shape_indices::ValidRow != DYNAMIC && tile_shape_indices::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(tile_shape_dst::Loc != Location::Acc &&
                tile_shape_src::Loc != Location::Acc &&
                tile_shape_indices::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static constexpr size_t row = tile_shape_dst::ValidRow;
  static constexpr size_t col = tile_shape_dst::ValidCol;
  if constexpr (tile_shape_src::isRowMajor &&
                tile_shape_indices::isRowMajor &&
                tile_shape_dst::isRowMajor) {
    TGather_Vec_RowMajor<tile_shape_dst, tile_shape_src, tile_shape_indices>
        <<<col, row, 1>>>(dst.data(), src.data(), indices.data());
  } else if constexpr (!tile_shape_src::isRowMajor &&
                       !tile_shape_indices::isRowMajor &&
                       !tile_shape_dst::isRowMajor) {
    TGather_Vec_ColMajor<tile_shape_dst, tile_shape_src, tile_shape_indices>
        <<<row, col, 1>>>(dst.data(), src.data(), indices.data());
  } else {
    static_assert(
        tile_shape_src::isRowMajor &&
            tile_shape_indices::isRowMajor &&
            tile_shape_dst::isRowMajor,
        "Storage type not supported");
  }
}

#endif