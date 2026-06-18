#ifndef TCOPYOUT_HPP
#define TCOPYOUT_HPP

#include "common/pto_tile.hpp"

using namespace pto;

// cube left -> gm row major
template <typename gm_shape, typename tile_shape>
void __mtc__ CopyOut2NzImpl1D(typename gm_shape::DType __out__ *dst,
                              const typename tile_shape::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  static constexpr size_t ceil_Rows =
      (tile_shape::ValidRow + tile_shape::InnerRows - 1) / tile_shape::InnerRows * tile_shape::InnerRows;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
    size_t idx_tile = sub_tile_i * ceil_Rows * inner_cols +
        sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
    size_t idx_gm = gm_shape::kRowStride * j + i;
    dst[idx_gm] = tile_ptr[idx_tile];
  }
}

template <typename gm_shape, typename tile_shape>
void __mtc__ CopyOut2ZnImpl1D(typename gm_shape::DType __out__ *dst,
                              const typename tile_shape::TileDType __in__ src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  static constexpr size_t ceil_cols =
      (tile_shape::ValidCol + tile_shape::InnerCols - 1) / tile_shape::InnerCols * tile_shape::InnerCols;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
    size_t sub_tile_j = j / inner_rows;
    size_t idx_j = j % inner_rows;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(src);
    size_t idx_gm = gm_shape::kRowStride * j + i;
    size_t index = sub_tile_j * ceil_cols * inner_rows +
        sub_tile_i * tile_shape::InnerNumel + idx_i * inner_rows + idx_j;
    dst[idx_gm] = tile_ptr[index];
  }
}

//no fractal
template <typename gm_shape, typename tile_shape>
void __mtc__
TCopyOut_Vec_ColMajor(typename gm_shape::DType __out__ *dst,
                      const typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::kColStride + i;
  size_t index_tile = j * tile_shape::ColStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}
 
template <typename gm_shape, typename tile_shape>
void __mtc__
TCopyOut_Vec_RowMajor(typename gm_shape::DType __out__ *dst,
                      typename tile_shape::TileDType __in__ src) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::kRowStride + i;
  size_t index_tile = j * tile_shape::RowStride + i;
  dst[index_gm] = blkv_get_tile_ptr(src)[index_tile];
}

template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TCOPYOUT_Impl(gm_shape &dst, tile_shape &src) {
  static constexpr size_t tile_rows = tile_shape::ValidRow;
  static constexpr size_t tile_cols = tile_shape::ValidCol;
  static_assert(tile_rows != DYNAMIC && tile_cols != DYNAMIC,
                "TODO: Support tile dynamic shape!");
  static_assert(gm_shape::staticStride[0] != DYNAMIC &&
                gm_shape::staticStride[1] != DYNAMIC &&
                gm_shape::staticStride[2] != DYNAMIC &&
                gm_shape::staticStride[3] != DYNAMIC &&
                gm_shape::staticStride[4] != DYNAMIC,
                "TODO: Support global tensor dynamic shape!");
  static_assert(gm_shape::staticStride[0] == 1 &&
                gm_shape::staticStride[1] == 1,
                "TODO: Support global tensor more than 3 dimensions");

#ifdef ENABLE_TENSOR_INSTR
  static constexpr size_t tile_all_cols = tile_shape::Cols;

  if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NZ2ND,
          dst.data(),
          gm_shape::kRowStride,
          src.data());
  } else if constexpr (is_Zn_layout<tile_shape>::value) { // Zn
       blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::ZN2ND,
          dst.data(),
          gm_shape::kRowStride,
          src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      if constexpr (gm_shape::isRowMajor) {
        blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NORM,
          dst.data(),
          gm_shape::kRowStride,
          src.data());
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (!tile_shape::isRowMajor) {
      if constexpr (!gm_shape::isRowMajor) {
        blk_tstore(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          LayoutCvtEnum::NORM,
          dst.data(),
          gm_shape::kColStride,
          src.data());
      } else {
        static_assert(!gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should colmajor");
      }
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false, 
                  "Storage layout type not supported");
  }

#else
  if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      CopyOut2NzImpl1D<gm_shape, tile_shape>
          <<<tile_cols, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (is_Zn_layout<tile_shape>::value) { // Zn
      CopyOut2ZnImpl1D<gm_shape, tile_shape>
          <<<tile_cols, 1, 1>>>(dst.data(), src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      if constexpr (gm_shape::isRowMajor) {
        TCopyOut_Vec_RowMajor<gm_shape, tile_shape>
            <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data());
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (!tile_shape::isRowMajor) {
      if constexpr (!gm_shape::isRowMajor) {
        TCopyOut_Vec_ColMajor<gm_shape, tile_shape>
            <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data());
      } else {
        static_assert(!gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should colmajor");
      }
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false, 
                  "Storage layout type not supported");
  }
#endif
}

#endif