#ifndef TCVT_HPP
#define TCVT_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TCvt_Nz2RowMajor_Imp(typename tile_shape_out::TileDType dst,
                          const typename tile_shape_in::TileDType src) {
  static constexpr int inner_rows = tile_shape_in::InnerRows;
  static constexpr int inner_cols = tile_shape_in::InnerCols;

  for (size_t i = 0; i < tile_shape_in::ValidCol; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;
    for (size_t j = 0; j < tile_shape_in::ValidRow; ++j) {
      size_t sub_tile_j = j / inner_rows;
      size_t idx_j = j % inner_rows;
      // sub_tile_j and idx_j can be replaced with a constant after loop
      // unrolling
      const typename tile_shape_in::DType *tile_ptr = src;
      typename tile_shape_out::DType data = *(
          tile_ptr + sub_tile_i * tile_shape_in::Rows * inner_cols +
          sub_tile_j * tile_shape_in::InnerNumel + idx_j * inner_cols + idx_i);
      dst[tile_shape_out::RowStride * j + i] = data;
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCvt_RowMajor2Nz_Imp(typename tile_shape_out::TileDType dst,
                          const typename tile_shape_in::TileDType src) {
  static constexpr int inner_rows = tile_shape_out::InnerRows;
  static constexpr int inner_cols = tile_shape_out::InnerCols;

  for (size_t i = 0; i < tile_shape_out::Cols; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;
    for (size_t j = 0; j < tile_shape_out::Rows; ++j) {
      size_t sub_tile_j = j / inner_rows;
      size_t idx_j = j % inner_rows;
      // sub_tile_j and idx_j can be replaced with a constant after loop
      // unrolling
      typename tile_shape_out::DType *tile_ptr = dst;
      typename tile_shape_in::DType data =
          *(src + tile_shape_in::RowStride * j + i);
      tile_ptr[sub_tile_i * tile_shape_out::Rows * inner_cols +
               sub_tile_j * tile_shape_out::InnerNumel + idx_j * inner_cols +
               idx_i] = data;
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCvt_Zn2Nz_Imp(typename tile_shape_out::TileDType dst,
                    const typename tile_shape_in::TileDType src) {

    // 原有逻辑
    for (size_t i = 0; i < tile_shape_out::ValidCol; ++i) {
        size_t out_sub_tile_i = i / tile_shape_out::InnerCols;
        size_t out_idx_i = i % tile_shape_out::InnerCols;
        size_t in_sub_tile_i = i / tile_shape_in::InnerCols;
        size_t in_idx_i = i % tile_shape_in::InnerCols;
        for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
            size_t out_sub_tile_j = j / tile_shape_out::InnerRows;
            size_t out_idx_j = j % tile_shape_out::InnerRows;
            size_t in_sub_tile_j = j / tile_shape_in::InnerRows;
            size_t in_idx_j = j % tile_shape_in::InnerRows;
            const typename tile_shape_in::DType *tile_ptr = src;
            typename tile_shape_out::DType data =
                *(tile_ptr + in_sub_tile_j * tile_shape_in::Cols * tile_shape_in::InnerRows +
                     in_sub_tile_i * tile_shape_in::InnerNumel + in_idx_i * tile_shape_in::InnerRows + in_idx_j);

            dst[out_sub_tile_i * tile_shape_out::Rows * tile_shape_out::InnerCols +
                    out_sub_tile_j * tile_shape_out::InnerNumel + out_idx_j * tile_shape_out::InnerCols +
                    out_idx_i] = data;
                    //DEBUG_PRINT_MSG("a[%d]=b[%d]=%f\n",a,b,data);
        }
    }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCvt_Nz2Zn_Imp(typename tile_shape_out::TileDType dst,
                    const typename tile_shape_in::TileDType src) {

    // 原有逻辑
    for (size_t i = 0; i < tile_shape_out::Cols; ++i) {
        size_t out_sub_tile_i = i / tile_shape_out::InnerCols;
        size_t in_sub_tile_i = i / tile_shape_in::InnerCols;
        size_t out_idx_i = i % tile_shape_out::InnerCols;
        size_t in_idx_i = i % tile_shape_in::InnerCols;
        for (size_t j = 0; j < tile_shape_out::Rows; ++j) {
            size_t out_sub_tile_j = j / tile_shape_out::InnerRows;
            size_t in_sub_tile_j = j / tile_shape_in::InnerRows;
            size_t out_idx_j = j % tile_shape_out::InnerRows;
            size_t in_idx_j = j % tile_shape_in::InnerRows;
            const typename tile_shape_in::DType *tile_ptr = src;
            typename tile_shape_out::DType data =
                *(tile_ptr + in_sub_tile_i * tile_shape_in::Rows * tile_shape_in::InnerCols +
                    in_sub_tile_j * tile_shape_in::InnerNumel + in_idx_j * tile_shape_in::InnerCols +
                    in_idx_i);

            dst[out_sub_tile_j * tile_shape_out::Cols * tile_shape_out::InnerRows +
                     out_sub_tile_i * tile_shape_out::InnerNumel + out_idx_i * tile_shape_out::InnerRows + out_idx_j] = data;
        }
    }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCvt_Nz2Nz_Imp(typename tile_shape_out::TileDType dst,
                    const typename tile_shape_in::TileDType src) {

    // 原有逻辑
    for (size_t i = 0; i < tile_shape_out::Cols; ++i) {
        size_t out_sub_tile_i = i / tile_shape_out::InnerCols;
        size_t in_sub_tile_i = i / tile_shape_in::InnerCols;
        size_t out_idx_i = i % tile_shape_out::InnerCols;
        size_t in_idx_i = i % tile_shape_in::InnerCols;
        for (size_t j = 0; j < tile_shape_out::Rows; ++j) {
            size_t out_sub_tile_j = j / tile_shape_out::InnerRows;
            size_t in_sub_tile_j = j / tile_shape_in::InnerRows;
            size_t out_idx_j = j % tile_shape_out::InnerRows;
            size_t in_idx_j = j % tile_shape_in::InnerRows;
            const typename tile_shape_in::DType *tile_ptr = src;
            typename tile_shape_out::DType data =
                *(tile_ptr + in_sub_tile_i * tile_shape_in::Rows * tile_shape_in::InnerCols +
                    in_sub_tile_j * tile_shape_in::InnerNumel + in_idx_j * tile_shape_in::InnerCols +
                    in_idx_i);

            dst[out_sub_tile_i * tile_shape_out::Rows * tile_shape_out::InnerCols +
                    out_sub_tile_j * tile_shape_out::InnerNumel + out_idx_j * tile_shape_out::InnerCols +
                    out_idx_i] = data;
        }
    }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCVT_Impl(tile_shape_out &dst, tile_shape_in &src) {
    if constexpr (is_Zn_layout<tile_shape_in>::value && is_Nz_layout<tile_shape_out>::value) {
        TCvt_Zn2Nz_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    }else if constexpr (is_Nz_layout<tile_shape_in>::value && is_Zn_layout<tile_shape_out>::value) {
        TCvt_Nz2Zn_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else if constexpr (tile_shape_in::isRowMajor && is_Nz_layout<tile_shape_out>::value) {
        TCvt_RowMajor2Nz_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else if constexpr (is_Nz_layout<tile_shape_in>::value && tile_shape_out::isRowMajor) {
        TCvt_Nz2RowMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else if constexpr (is_Nz_layout<tile_shape_in>::value && is_Nz_layout<tile_shape_out>::value) {
        TCvt_Nz2Nz_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else {
        static_assert(tile_shape_in::isRowMajor || tile_shape_in::isRowMajor, "Storage layout type not supported");
    }
}

#endif