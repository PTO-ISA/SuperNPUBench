#ifndef TCOPYIN_HPP
#define TCOPYIN_HPP

#include "common/pto_tile.hpp"
#include "template_asm.hpp"

using namespace pto;

// gm row major -> tile Nz
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInRow2NzImpl1D(typename tile_shape::TileDType __out__ dst,
                                const typename gm_shape::DType __in__ *src) {
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
    __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
    size_t idx_gm = gm_shape::RowStride * j + i;
    size_t index = sub_tile_i * ceil_Rows * inner_cols +
        sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
    tile_ptr[index] = src[idx_gm];
  }
}

// gm col major -> tile Zn
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInCol2ZnImpl1D(typename tile_shape::TileDType __out__ dst,
                                const typename gm_shape::DType __in__ *src) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  static constexpr size_t ceil_cols =
      (tile_shape::ValidCol + tile_shape::InnerCols - 1) / tile_shape::InnerCols * tile_shape::InnerCols;
  size_t i = blkv_get_index_x();

  size_t sub_tile_i = i / inner_rows;
  size_t idx_i = i % inner_rows;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
    size_t sub_tile_j = j / inner_cols;
    size_t idx_j = j % inner_cols;
    // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
    __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
    size_t idx_gm = gm_shape::ColStride * j + i;
    size_t index = sub_tile_i * ceil_cols * inner_rows +
        sub_tile_j * tile_shape::InnerNumel + idx_j * inner_rows + idx_i;
    tile_ptr[index] = src[idx_gm];
  }
}

// gm row major -> tile Zn
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInRow2ZnImpl1D(typename tile_shape::TileDType __out__ dst,
                                const typename gm_shape::DType __in__ *src) {
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
    __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
    size_t idx_gm = gm_shape::RowStride * j + i;
    size_t index = sub_tile_j * ceil_cols * inner_rows +
        sub_tile_i * tile_shape::InnerNumel + idx_i * inner_rows + idx_j;
    tile_ptr[index] = src[idx_gm];
  }
}

//no fractal
template <typename tile_shape, typename gm_shape>
void __mtc__ TCopyIn_Vec_ColMajor(typename tile_shape::TileDType __out__ dst,
                                  const typename gm_shape::DType __in__ *src) {
 
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::ColStride + i;
  size_t index_tile = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index_tile] = src[index_gm];
}
 
template <typename tile_shape, typename gm_shape>
void __mtc__ TCopyIn_Vec_RowMajor(typename tile_shape::TileDType __out__ dst,
                                  typename gm_shape::DType __in__ *src) {
 
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_shape::RowStride + i;
  size_t index_tile = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index_tile] = src[index_gm];
}

// gm row major -> tile Nz
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInRow2NzImpl2D_Dynamic(typename tile_shape::TileDType __out__ dst,
                                        const typename gm_shape::DType __in__ *src,
                                        const size_t __in__ gm_row_stride) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
  size_t idx_gm = gm_row_stride * j + i;
  size_t index = sub_tile_i * tile_shape::Rows * inner_cols +
      sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
  tile_ptr[index] = src[idx_gm];
}

// gm col major -> tile Zn
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInCol2ZnImpl2D_Dynamic(typename tile_shape::TileDType __out__ dst,
                                        const typename gm_shape::DType __in__ *src,
                                        const size_t __in__ gm_col_stride) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_rows;
  size_t idx_i = i % inner_rows;

  size_t sub_tile_j = j / inner_cols;
  size_t idx_j = j % inner_cols;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
  size_t idx_gm = gm_col_stride * j + i;
  size_t index = sub_tile_i * tile_shape::Cols * inner_rows +
      sub_tile_j * tile_shape::InnerNumel + idx_j * inner_rows + idx_i;
  tile_ptr[index] = src[idx_gm];
}

// gm row major -> tile Zn
template <typename tile_shape, typename gm_shape>
void __mtc__ CopyInRow2ZnImpl2D_Dynamic(typename tile_shape::TileDType __out__ dst,
                                        const typename gm_shape::DType __in__ *src,
                                        const size_t __in__ gm_row_stride) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t sub_tile_i = i / inner_cols;
  size_t idx_i = i % inner_cols;

  size_t sub_tile_j = j / inner_rows;
  size_t idx_j = j % inner_rows;
  // sub_tile_j and idx_j can be replaced with a constant after loop unrolling
  __vbuf__ typename tile_shape::DType *tile_ptr = blkv_get_tile_ptr(dst);
  size_t idx_gm = gm_row_stride * j + i;
  size_t index = sub_tile_j * tile_shape::Cols * inner_rows +
      sub_tile_i * tile_shape::InnerNumel + idx_i * inner_rows + idx_j;
  tile_ptr[index] = src[idx_gm];
}

//no fractal
template <typename tile_shape, typename gm_shape>
void __mtc__ TCopyIn_Vec_ColMajor_Dynamic(typename tile_shape::TileDType __out__ dst,
                                          const typename gm_shape::DType __in__ *src,
                                          const size_t __in__ gm_col_stride) {
 
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_col_stride + i;
  size_t index_tile = j * tile_shape::ColStride + i;
  blkv_get_tile_ptr(dst)[index_tile] = src[index_gm];
}

template <typename tile_shape, typename gm_shape>
void __mtc__ TCopyIn_Vec_RowMajor_Dynamic(typename tile_shape::TileDType __out__ dst,
                                          typename gm_shape::DType __in__ *src,
                                          const size_t __in__ gm_row_stride) {
 
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  size_t index_gm = j * gm_row_stride + i;
  size_t index_tile = j * tile_shape::RowStride + i;
  blkv_get_tile_ptr(dst)[index_tile] = src[index_gm];
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void _TCOPYIN_Impl(tile_shape &dst, gm_shape &src) {
  size_t tile_rows = dst.GetValidRow();
  size_t tile_cols = dst.GetValidCol();
  static_assert(tile_shape::Loc != Location::Acc, "Unsupport ACC to be input or output here");
  static_assert(gm_shape::staticStride[0] == 1 &&
                gm_shape::staticStride[1] == 1,
                "TODO: Support global tensor more than 3 dimensions");

#ifdef ENABLE_TENSOR_INSTR
  static_assert(!tile_shape::isBoxedLayout || tile_shape::SFractalSize == 512,
                    "Error! Cude blk_tload:FractalSize != 512");
  static constexpr size_t tile_all_cols = tile_shape::Cols;
  static constexpr size_t tile_all_rows = tile_shape::Rows;
  static constexpr unsigned Pad = (unsigned)PadValue::Null;
  if constexpr (is_Nz_layout<tile_shape>::value) {
    if constexpr (gm_shape::isRowMajor) {
      blk_tload(tile_cols, tile_rows, tile_all_cols,
          type_traits<typename tile_shape::DType>::TypeCode,
          Pad,
          LayoutCvtEnum::ND2NZ,
          dst.data(),
          src.data(),
          (src.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8);
    } else {
      static_assert(gm_shape::isRowMajor,
                    "Storage layout type not supported, gm should rowmajor");
    }
  } else if constexpr (is_Zn_layout<tile_shape>::value) {
    if constexpr (!gm_shape::isRowMajor) {
      blk_tload(tile_cols, tile_rows, tile_all_cols,
            type_traits<typename tile_shape::DType>::TypeCode,
            Pad,
            LayoutCvtEnum::DN2ZN,
            dst.data(),
            src.data(),
            (src.GetStride(4) * type_traits<typename gm_shape::DType>::bits + 7) / 8);
    } else {
      blk_tload(tile_cols, tile_rows, tile_all_cols,
            type_traits<typename tile_shape::DType>::TypeCode,
            Pad,
            LayoutCvtEnum::ND2ZN,
            dst.data(),
            src.data(),
            (src.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8);
    }
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      if constexpr (gm_shape::isRowMajor) {
        blk_tload(tile_cols, tile_rows, tile_all_cols,
            type_traits<typename tile_shape::DType>::TypeCode,
            Pad,
            LayoutCvtEnum::NORM,
            dst.data(),
            src.data(),
            (src.GetStride(3) * type_traits<typename gm_shape::DType>::bits + 7) / 8);
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (!tile_shape::isRowMajor) {
      if constexpr (!gm_shape::isRowMajor) {
        blk_tload(tile_rows, tile_cols, tile_all_rows,
            type_traits<typename tile_shape::DType>::TypeCode,
            Pad,
            LayoutCvtEnum::NORM,
            dst.data(),
            src.data(),
            (src.GetStride(4) * type_traits<typename gm_shape::DType>::bits + 7) / 8);
      } else {
        static_assert(!gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should colmajor");
      }
    } else {
        static_assert(tile_shape::isBoxedLayout == false,
                      "Storage layout type not supported");
    }
  }

#else
  // TODO: support static tile && dynamic gm
  if constexpr (gm_shape::RowStride == DYNAMIC || gm_shape::ColStride == DYNAMIC ||
                tile_shape::ValidRow == DYNAMIC || tile_shape::ValidCol == DYNAMIC) { // dynamic
    if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      if constexpr (gm_shape::isRowMajor) {
        CopyInRow2NzImpl2D_Dynamic<tile_shape, gm_shape>
            <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), src.GetStride(3));
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (is_Zn_layout<tile_shape>::value) { //Zn
      if constexpr (!gm_shape::isRowMajor) {
        CopyInCol2ZnImpl2D_Dynamic<tile_shape, gm_shape>
            <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data(), src.GetStride(4));
      } else {
        CopyInRow2ZnImpl2D_Dynamic<tile_shape, gm_shape>
            <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), src.GetStride(3));
      }
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        if constexpr (gm_shape::isRowMajor) {
          TCopyIn_Vec_RowMajor_Dynamic<tile_shape, gm_shape>
              <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data(), src.GetStride(3));
        } else {
          static_assert(gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should rowmajor");
        }
      } else if constexpr (!tile_shape::isRowMajor) {
        if constexpr (!gm_shape::isRowMajor) {
          TCopyIn_Vec_ColMajor_Dynamic<tile_shape, gm_shape>
              <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data(), src.GetStride(4));
        } else {
          static_assert(!gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should colmajor");
        }
      } else {
          static_assert(tile_shape::isBoxedLayout == false,
                        "Storage layout type not supported");
      }
    }
  } else { // static
    if constexpr (is_Nz_layout<tile_shape>::value) { // Nz
      if constexpr (gm_shape::isRowMajor) {
        CopyInRow2NzImpl1D<tile_shape, gm_shape>
            <<<tile_cols, 1, 1>>>(dst.data(), src.data());
      } else {
        static_assert(gm_shape::isRowMajor,
                      "Storage layout type not supported, gm should rowmajor");
      }
    } else if constexpr (is_Zn_layout<tile_shape>::value) { //Zn
      if constexpr (!gm_shape::isRowMajor) {
        CopyInCol2ZnImpl1D<tile_shape, gm_shape>
            <<<tile_rows, 1, 1>>>(dst.data(), src.data());
      } else {
        CopyInRow2ZnImpl1D<tile_shape, gm_shape>
            <<<tile_cols, 1, 1>>>(dst.data(), src.data());
      }
    } else if constexpr (tile_shape::isBoxedLayout == false) {
      if constexpr (tile_shape::isRowMajor) {
        if constexpr (gm_shape::isRowMajor) {
          TCopyIn_Vec_RowMajor<tile_shape, gm_shape>
              <<<tile_cols, tile_rows, 1>>>(dst.data(), src.data());
        } else {
          static_assert(gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should rowmajor");
        }
      } else if constexpr (!tile_shape::isRowMajor) {
        if constexpr (!gm_shape::isRowMajor) {
          TCopyIn_Vec_ColMajor<tile_shape, gm_shape>
              <<<tile_rows, tile_cols, 1>>>(dst.data(), src.data());
        } else {
          static_assert(!gm_shape::isRowMajor,
                        "Storage layout type not supported, gm should colmajor");
        }
      } else {
          static_assert(tile_shape::isBoxedLayout == false,
                        "Storage layout type not supported");
      }
    }
  }

  
#endif
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TCOPYIN_2LVL(tile_shape &dst, gm_shape &src){
  using tile_tmp = Tile<Location::Vec, typename gm_shape::DType, tile_shape::Rows, tile_shape::Cols, gm_shape::isRowMajor? BLayout::RowMajor:BLayout::ColMajor, tile_shape::ValidRow, tile_shape::ValidCol>;
  tile_tmp tmp;
  _TCOPYIN_Impl(tmp, src);
  if constexpr(gm_shape::isRowMajor && is_Nz_layout<tile_shape>::value){
    TCVT_ND2NZ(dst, tmp);    
  }else if constexpr(gm_shape::isRowMajor && is_Zn_layout<tile_shape>::value){
    TCVT_ND2ZN(dst, tmp); 
  }else if constexpr(!gm_shape::isRowMajor && is_Nz_layout<tile_shape>::value){
    TCVT_DN2NZ(dst, tmp);
  }else if constexpr(!gm_shape::isRowMajor && is_Zn_layout<tile_shape>::value){
    TCVT_DN2ZN(dst, tmp);
  }
}

template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TCOPYIN_Impl(tile_shape &dst, gm_shape &src) {
  #ifdef RUMINATE
  TCOPYIN_2LVL(dst, src);
  #else
  _TCOPYIN_Impl(dst, src);
  #endif
}

#endif