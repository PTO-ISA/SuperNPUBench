#ifndef TEXPANDSCALAR_HPP
#define TEXPANDSCALAR_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape>
void __vec__
ExpandScalarImpl_RowMajor(typename tile_shape::TileDType __out__ dst,
                          const typename tile_shape::DType __in__ val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::RowStride;
  blkv_get_tile_ptr(dst)[index] = val;
}

template <typename tile_shape>
void __vec__
ExpandScalarImpl_ColMajor(typename tile_shape::TileDType __out__ dst,
                          const typename tile_shape::DType __in__ val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = i + j * tile_shape::ColStride;
  blkv_get_tile_ptr(dst)[index] = val;
}
template <typename tile_shape>
void __vec__ ExpandScalar2NzImpl(typename tile_shape::TileDType __out__ dst,
                                 const typename tile_shape::DType __in__ val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int col_fract_nums =
      tile_shape::Cols / tile_shape::InnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < col_fract_nums; k++) {
    size_t index =
        k * tile_shape::Rows * tile_shape::InnerCols + j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = val;
  }
}

template <typename tile_shape>
void __vec__ ExpandScalar2ZnImpl(typename tile_shape::TileDType __out__ dst,
                                 const typename tile_shape::DType __in__ val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  static constexpr int row_fract_nums =
      tile_shape::Rows / tile_shape::InnerRows;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < row_fract_nums; k++) {
    size_t index =
        k * tile_shape::Cols * tile_shape::InnerRows + j * LaneNum + i;
    blkv_get_tile_ptr(dst)[index] = val;
  }
}

template <typename tile_shape>
void ExpandScalar2NzImpl_Dynamic(typename tile_shape::TileDType dst,
                         const typename tile_shape::DType val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;
  size_t index = sub_tile_i * tile_shape::Rows * inner_cols +
                  sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols +
                  idx_i;
  blkv_get_tile_ptr(dst)[index] = val;
}

template <typename tile_shape>
void ExpandScalar2ZnImpl_Dynamic(typename tile_shape::TileDType dst,
                         const typename tile_shape::DType val) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  unsigned sub_tile_i = i / inner_rows;
  unsigned idx_i = i % inner_rows;

  unsigned sub_tile_j = j / inner_cols;
  unsigned idx_j = j % inner_cols;
  unsigned index = sub_tile_i * tile_shape::Cols * inner_rows +
                    sub_tile_j * tile_shape::InnerNumel +
                    idx_j * inner_rows + idx_i;
  blkv_get_tile_ptr(dst)[index] = val;
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_Impl(tile_shape &dst, typename tile_shape::DType s) {
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  size_t row = dst.GetValidRow();
  size_t col = dst.GetValidCol();

  if constexpr (tile_shape::ValidRow == DYNAMIC || tile_shape::ValidCol == DYNAMIC) { // dynamic
    if constexpr (is_Nz_layout<tile_shape>::value) {
      ExpandScalar2NzImpl_Dynamic<tile_shape><<<col, row, 1>>>(dst.data(), s);
    } else if constexpr (is_Zn_layout<tile_shape>::value) {
      ExpandScalar2ZnImpl_Dynamic<tile_shape><<<row, col, 1>>>(dst.data(), s);
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        ExpandScalarImpl_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), s);
      } else {
        ExpandScalarImpl_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), s);
      }
    } else {
      static_assert(is_Nz_layout<tile_shape>::value &&
                        tile_shape::isBoxedLayout == false,
                    "Storage type not supported");
    }
  } else { // static
    if constexpr (is_Nz_layout<tile_shape>::value) {
      static constexpr size_t Y =
        tile_shape::Rows / (LaneNum / tile_shape::InnerCols);
      ExpandScalar2NzImpl<tile_shape><<<LaneNum, Y, 1>>>(dst.data(), s);
    } else if constexpr (is_Zn_layout<tile_shape>::value) {
      static constexpr size_t Y =
        tile_shape::Cols / (LaneNum / tile_shape::InnerRows);
      ExpandScalar2ZnImpl<tile_shape><<<LaneNum, Y, 1>>>(dst.data(), s);
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        ExpandScalarImpl_RowMajor<tile_shape><<<col, row, 1>>>(dst.data(), s);
      } else {
        ExpandScalarImpl_ColMajor<tile_shape><<<row, col, 1>>>(dst.data(), s);
      }
    } else {
      static_assert(is_Nz_layout<tile_shape>::value &&
                        tile_shape::isBoxedLayout == false,
                    "Storage type not supported");
    }
  }
}

#endif

