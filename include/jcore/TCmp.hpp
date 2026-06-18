#ifndef TCMP_HPP
#define TCMP_HPP

#include "common/pto_tile.hpp"
#include "jcore/constants.hpp"
#include <assert.h>
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in, CmpMode mode>
void __vec__ TCmp_Vec_RowMajor(typename tile_shape_out::TileDType __out__ dst,
                                const typename tile_shape_in::TileDType __in__ src0,
                                const typename tile_shape_in::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = j * tile_shape_in::RowStride + i;

  typename tile_shape_in::DType a = blkv_get_tile_ptr(src0)[index];
  typename tile_shape_in::DType b = blkv_get_tile_ptr(src1)[index];
  typename tile_shape_out::DType result = 0;
  if constexpr (mode == CmpMode::EQ) {
      result = static_cast<typename tile_shape_out::DType>( a == b);
  } else if constexpr (mode == CmpMode::NE) {
      result = static_cast<typename tile_shape_out::DType>( a != b);
  } else if constexpr (mode == CmpMode::GT) {
      result = static_cast<typename tile_shape_out::DType>( a > b);
  } else if constexpr (mode == CmpMode::LT) {
      result = static_cast<typename tile_shape_out::DType>( a < b);
  } else if constexpr (mode == CmpMode::GE) {
      result = static_cast<typename tile_shape_out::DType>( a >= b);
  } else if constexpr (mode == CmpMode::LE) {
      result = static_cast<typename tile_shape_out::DType>( a <= b);
  }
  blkv_get_tile_ptr(dst)[index] = result;
}

template <typename tile_shape_out, typename tile_shape_in, CmpMode mode>
void __vec__ TCmp_Vec_ColMajor(typename tile_shape_out::TileDType __out__ dst,
                                const typename tile_shape_in::TileDType __in__ src0,
                                const typename tile_shape_in::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  size_t index = j * tile_shape_in::ColStride + i;

  typename tile_shape_in::DType a = blkv_get_tile_ptr(src0)[index];
  typename tile_shape_in::DType b = blkv_get_tile_ptr(src1)[index];
  typename tile_shape_out::DType result = 0;
  if constexpr (mode == CmpMode::EQ) {
      result = static_cast<typename tile_shape_out::DType>( a == b);
  } else if constexpr (mode == CmpMode::NE) {
      result = static_cast<typename tile_shape_out::DType>( a != b);
  } else if constexpr (mode == CmpMode::GT) {
      result = static_cast<typename tile_shape_out::DType>( a > b);
  } else if constexpr (mode == CmpMode::LT) {
      result = static_cast<typename tile_shape_out::DType>( a < b);
  } else if constexpr (mode == CmpMode::GE) {
      result = static_cast<typename tile_shape_out::DType>( a >= b);
  } else if constexpr (mode == CmpMode::LE) {
      result = static_cast<typename tile_shape_out::DType>( a <= b);
  }
  blkv_get_tile_ptr(dst)[index] = result;
}

template <typename tile_shape_out, typename tile_shape_in>
void TCMP_Impl(tile_shape_out &dst, tile_shape_in &src0, tile_shape_in &src1, CmpMode cmpMode) {
  static_assert(tile_shape_in::Rows == tile_shape_out::Rows &&
                    tile_shape_in::Cols == tile_shape_out::Cols,
                "Error! Input shape != Output shape");
  static_assert(tile_shape_in::InnerRows == tile_shape_out::InnerRows &&
                    tile_shape_in::InnerCols == tile_shape_out::InnerCols,
                "Error! Inner shape is not equal!");
  static_assert(tile_shape_out::Loc == Location::Vec && 
                tile_shape_in::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(tile_shape_out::isBoxedLayout == false && 
                tile_shape_in::isBoxedLayout == false, "TCMP not support Boxed Layout!");

  static constexpr size_t row = tile_shape_in::ValidRow;
  static constexpr size_t col = tile_shape_in::ValidCol;
  static_assert(row != DYNAMIC && col != DYNAMIC,
                "TODO: Support tile dynamic shape!");

  if constexpr (std::is_same<typename tile_shape_in::DType, int64_t>::value ||
                std::is_same<typename tile_shape_in::DType, int32_t>::value ||
                std::is_same<typename tile_shape_in::DType, float>::value ||
                std::is_same<typename tile_shape_in::DType, __half>::value) {
    if constexpr (tile_shape_in::isRowMajor) {
      switch (cmpMode) {
        case CmpMode::EQ:
            if (std::is_same<typename tile_shape_in::DType, int32_t>::value) {
                TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::EQ>
                    <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            } else {
                assert(false && "Only Int32_t type supports the EQ mode!");
            }
            break;
        case CmpMode::NE:
            TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::NE>
                <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::GT:
            TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::GT>
                <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::LT:
            TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::LT>
                <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::GE:
            TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::GE>
                <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::LE:
            TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in, CmpMode::LE>
                <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        default:
            assert(false && "Unsupported CmpMode value");
            break;
      }
    } else {
      switch (cmpMode) {
        case CmpMode::EQ:
            if (std::is_same<typename tile_shape_in::DType, int32_t>::value) {
                TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::EQ>
                    <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            } else {
                assert(false && "Only Int32_t type supports the EQ mode!");
            }
            break;
        case CmpMode::NE:
            TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::NE>
                <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::GT:
            TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::GT>
                <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::LT:
            TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::LT>
                <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::GE:
            TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::GE>
                <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        case CmpMode::LE:
            TCmp_Vec_ColMajor<tile_shape_out, tile_shape_in, CmpMode::LE>
                <<<row, col, 1>>>(dst.data(), src0.data(), src1.data());
            break;
        default:
            assert(false && "Unsupported CmpMode value");
            break;
      }
    }
  } else {
    static_assert(std::is_same<typename tile_shape_in::DType, int64_t>::value ||
                    std::is_same<typename tile_shape_in::DType, int32_t>::value ||
                    std::is_same<typename tile_shape_in::DType, float>::value ||
                    std::is_same<typename tile_shape_in::DType, __half>::value,
                    "Dtype not support!");
  }
}
#endif
