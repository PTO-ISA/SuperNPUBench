#ifndef TASSEMBLE_HPP
#define TASSEMBLE_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void __vec__
TAssemble_RowMajor_Imp(typename tile_shape_out::TileDType __out__ dst,
                       const typename tile_shape_in0::TileDType __in__ src0,
                       const typename tile_shape_in1::TileDType __in__ src1,
                       const typename tile_shape_in2::TileDType __in__ src2) {
  size_t i = blkv_get_index_x();
  typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in0::DType *src0_ptr = blkv_get_tile_ptr(src0);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
    size_t idx_out0 = i * tile_shape_out::RowStride + j;
    size_t idx_in0 = i * tile_shape_in0::RowStride + j;
    dst_ptr[idx_out0] = src0_ptr[idx_in0];
  }

  typename tile_shape_in1::DType *src1_ptr = blkv_get_tile_ptr(src1);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
    size_t idx_out1 =
        tile_shape_in0::ValidCol + i * tile_shape_out::RowStride + j;
    size_t idx_in1 = i * tile_shape_in1::RowStride + j;
    dst_ptr[idx_out1] = src1_ptr[idx_in1];
  }

  typename tile_shape_in2::DType *src2_ptr = blkv_get_tile_ptr(src2);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in2::ValidCol; ++j) {
    size_t idx_out2 = tile_shape_in0::ValidCol + tile_shape_in1::ValidCol +
                      i * tile_shape_out::RowStride + j;
    size_t idx_in2 = i * tile_shape_in2::RowStride + j;
    dst_ptr[idx_out2] = src2_ptr[idx_in2];
  }
}

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void __vec__
TAssemble_ColMajor_Imp(typename tile_shape_out::TileDType __out__ dst,
                       const typename tile_shape_in0::TileDType __in__ src0,
                       const typename tile_shape_in1::TileDType __in__ src1,
                       const typename tile_shape_in2::TileDType __in__ src2) {
  size_t i = blkv_get_index_x();
  typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in0::DType *src0_ptr = blkv_get_tile_ptr(src0);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
    size_t idx_out0 = j * tile_shape_out::ColStride + i;
    size_t idx_in0 = j * tile_shape_in0::ColStride + i;
    dst_ptr[idx_out0] = src0_ptr[idx_in0];
  }

  typename tile_shape_in1::DType *src1_ptr = blkv_get_tile_ptr(src1);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
    size_t idx_out1 =
        tile_shape_in0::Numel + j * tile_shape_out::ColStride + i;
    size_t idx_in1 = j * tile_shape_in1::ColStride + i;
    dst_ptr[idx_out1] = src1_ptr[idx_in1];
  }

  typename tile_shape_in2::DType *src2_ptr = blkv_get_tile_ptr(src2);
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape_in2::ValidCol; ++j) {
    size_t idx_out2 = tile_shape_in0::Numel + tile_shape_in1::Numel +
                      j * tile_shape_out::ColStride + i;
    size_t idx_in2 = j * tile_shape_in2::ColStride + i;
    dst_ptr[idx_out2] = src2_ptr[idx_in2];
  }
}

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void __vec__ TAssemble_NzLayout_Imp(
    typename tile_shape_out::TileDType __out__ dst,
    const typename tile_shape_in0::TileDType __in__ src0,
    const typename tile_shape_in1::TileDType __in__ src1,
    const typename tile_shape_in2::TileDType __in__ src2) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  typename tile_shape_out::DType *dst_ptr = blkv_get_tile_ptr(dst);
  typename tile_shape_in0::DType *src0_ptr = blkv_get_tile_ptr(src0);
  static constexpr int block_cols_src0 =
      tile_shape_in0::Cols / tile_shape_in0::kInnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols_src0; ++k) {
    size_t idx_in0 = k * tile_shape_in0::Rows * tile_shape_in0::kInnerCols +
                     j * LaneNum + i;
    size_t idx_out0 = idx_in0;
    dst_ptr[idx_out0] = src0_ptr[idx_in0];
  }

  typename tile_shape_in1::DType *src1_ptr = blkv_get_tile_ptr(src1);
  static constexpr int block_cols_src1 =
      tile_shape_in1::Cols / tile_shape_in1::kInnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols_src1; ++k) {
    size_t idx_in1 = k * tile_shape_in1::Rows * tile_shape_in1::kInnerCols +
                     j * LaneNum + i;
    size_t idx_out1 = tile_shape_in0::Numel + idx_in1;
    dst_ptr[idx_out1] = src1_ptr[idx_in1];
  }

  typename tile_shape_in2::DType *src2_ptr = blkv_get_tile_ptr(src2);
  static constexpr int block_cols_src2 =
      tile_shape_in2::Cols / tile_shape_in2::kInnerCols;
#pragma clang loop unroll(full)
  for (size_t k = 0; k < block_cols_src2; ++k) {
    size_t idx_in2 = k * tile_shape_in1::Rows * tile_shape_in1::kInnerCols +
                     j * LaneNum + i;
    size_t idx_out2 = tile_shape_in0::Numel + tile_shape_in1::Numel + idx_in2;
    dst_ptr[idx_out2] = src2_ptr[idx_in2];
  }
}

template <is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1,
          is_tile_data_v tile_shape_in2, is_tile_data_v tile_shape_out>
void TASSEMBLE_Impl(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1,
               tile_shape_in2 &src2) {
  static_assert(tile_shape_out::Rows == tile_shape_in0::Rows &&
                    tile_shape_out::Rows == tile_shape_in1::Rows &&
                    tile_shape_out::Rows == tile_shape_in2::Rows,
                "Error! All tile rows must be equal");
  static_assert(tile_shape_in0::Cols == tile_shape_in0::ValidCol &&
                    tile_shape_in1::Cols == tile_shape_in1::ValidCol &&
                    tile_shape_in2::Cols == tile_shape_in2::ValidCol,
                "Error! Mask for columns are forbidden");
  static_assert(
      tile_shape_out::Cols ==
          tile_shape_in0::Cols + tile_shape_in1::Cols + tile_shape_in2::Cols,
      "Error! Output columns must equal the sum of the input columns.");

  static constexpr size_t dst_row = tile_shape_out::ValidRow;
  static constexpr size_t row_lines =
      tile_shape_out::Rows / (LaneNum / tile_shape_out::InnerCols);
  if constexpr (is_Nz_layout<tile_shape_out>::value &&
                is_Nz_layout<tile_shape_in0>::value &&
                is_Nz_layout<tile_shape_in1>::value &&
                is_Nz_layout<tile_shape_in2>::value) {
    TAssemble_NzLayout_Imp<tile_shape_in0, tile_shape_in1,
                                    tile_shape_in2, tile_shape_out>
        <<<LaneNum, row_lines, 1>>>(dst.data(), src0.data(), src1.data(),
                                    src2.data());
  } else if constexpr (tile_shape_in0::isBoxedLayout == false &&
                       tile_shape_in1::isBoxedLayout == false &&
                       tile_shape_in2::isBoxedLayout == false &&
                       tile_shape_out::isBoxedLayout == false) {
    if constexpr (tile_shape_in0::isRowMajor &&
                  tile_shape_in1::isRowMajor &&
                  tile_shape_in2::isRowMajor &&
                  tile_shape_out::isRowMajor) {
      TAssemble_RowMajor_Imp<tile_shape_in0, tile_shape_in1, tile_shape_in2,
                             tile_shape_out><<<dst_row, 1, 1>>>(
          dst.data(), src0.data(), src1.data(), src2.data());
    } else if constexpr (!tile_shape_in0::isRowMajor &&
                         !tile_shape_in1::isRowMajor &&
                         !tile_shape_in2::isRowMajor &&
                         !tile_shape_out::isRowMajor) {
      TAssemble_ColMajor_Imp<tile_shape_in0, tile_shape_in1, tile_shape_in2,
                             tile_shape_out><<<dst_row, 1, 1>>>(
          dst.data(), src0.data(), src1.data(), src2.data());
    } else {
      static_assert(tile_shape_in0::isRowMajor &&
                        tile_shape_in1::isRowMajor &&
                        tile_shape_in2::isRowMajor &&
                        tile_shape_out::isRowMajor,
                    "Storage layout type not supported");
    }
  } else {
    static_assert(tile_shape_in0::isBoxedLayout == false &&
                      tile_shape_in1::isBoxedLayout == false &&
                      tile_shape_in2::isBoxedLayout == false &&
                      tile_shape_out::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif