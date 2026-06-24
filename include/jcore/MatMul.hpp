#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "common/pto_tile.hpp"

using namespace pto;

#ifdef __linx

// Direct-boot Linx smoke uses the scalar fallback until the vector launch
// syntax is supported in this toolchain lane.
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_C>
void MATMUL_Impl(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  static_assert(!tile_shape_A::isBoxedLayout && !tile_shape_B::isBoxedLayout &&
                    !tile_shape_C::isBoxedLayout,
                "Linx scalar MATMUL supports only unboxed layouts");
  static_assert(tile_shape_A::Loc != Location::Acc &&
                    tile_shape_B::Loc != Location::Acc &&
                    tile_shape_C::Loc != Location::Acc,
                "Linx scalar MATMUL does not support ACC tile operands");

  const size_t rows = dst.GetValidRow();
  const size_t cols = dst.GetValidCol();
  const size_t inner =
      src0.GetValidCol() > src1.GetValidRow() ? src0.GetValidCol()
                                               : src1.GetValidRow();

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      typename tile_shape_C::DType acc = 0;
      for (size_t k = 0; k < inner; ++k) {
        if constexpr (!std::is_same<typename tile_shape_A::DType, __half>::value &&
                      !std::is_same<typename tile_shape_B::DType, __half>::value &&
                      !std::is_same<typename tile_shape_C::DType, __half>::value) {
          acc += src0.data()[index<tile_shape_A>(row, k)] *
                 src1.data()[index<tile_shape_B>(k, col)];
        }
      }
      dst.data()[index<tile_shape_C>(row, col)] = acc;
    }
  }
}

#else

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void __vec__ MatMul_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_B::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  typename tile_shape_C::DType data = 0;

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < tile_shape_A::ValidCol; ++k){
    size_t idx_0 = index<tile_shape_A>(j, k);
    size_t idx_1 = index<tile_shape_B>(k, i);
    data += blkv_get_tile_ptr(src0)[idx_0] * 
            blkv_get_tile_ptr(src1)[idx_1];
  }
  size_t idx = index<tile_shape_C>(j, i);
  blkv_get_tile_ptr(dst)[idx] = data;
}

// Matrix Multiply: C[MxN] = A[M×K] x B[KxN]
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_C>
void MATMUL_Impl(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  // static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
  //               "Error! Cude A:Columns != Cude B:Rows");

  size_t M = dst.GetValidRow();
  size_t N = dst.GetValidCol();
  size_t K =
    (src0.GetValidCol() > src1.GetValidRow())
        ? src0.GetValidCol()
        : src1.GetValidRow();

    if constexpr ((!tile_shape_A::isBoxedLayout) && 
                  (!tile_shape_B::isBoxedLayout) &&
                  (!tile_shape_C::isBoxedLayout)) {
      MatMul_Vec_Impl<tile_shape_A, tile_shape_B, tile_shape_C>
      <<<N, M, 1>>>(dst.data(), src0.data(), src1.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cude C:FractalSize != 1024 or not Nz_layout");
      static_assert(tile_shape_A::Loc != Location::Acc && tile_shape_B::Loc != Location::Acc, "Error! Matmul input can not be ACC");
      static_assert(tile_shape_C::Loc == Location::Acc, "Error! Matmul output only canbe ACC");
      blk_matmul(M, N, K, type_traits<typename tile_shape_A::DType>::TypeCode, type_traits<typename tile_shape_B::DType>::TypeCode,
                 dst.data(), src0.data(), src1.data());
    } else {
      static_assert((!tile_shape_A::isBoxedLayout) && 
                    (!tile_shape_B::isBoxedLayout), 
                    "Storage layout type not supported");
    }
}

template <typename tile_shape_A, typename tile_shape_AX,
          typename tile_shape_B, typename tile_shape_BX,
          typename tile_shape_C>
void __vec__ MatMulMx_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_AX::TileDType __in__ src0_scale,
    const typename tile_shape_B::TileDType __in__ src1,
    const typename tile_shape_BX::TileDType __in__ src1_scale) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  typename tile_shape_C::DType data = 0;

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < tile_shape_A::ValidCol; ++k) {
    size_t idx_0  = index<tile_shape_A>(j, k);
    size_t idx_0x = index<tile_shape_AX>(j, k);
    size_t idx_1  = index<tile_shape_B>(k, i);
    size_t idx_1x = index<tile_shape_BX>(k, i);

    data += (blkv_get_tile_ptr(src0)[idx_0] *
             blkv_get_tile_ptr(src0_scale)[idx_0x]) *
            (blkv_get_tile_ptr(src1)[idx_1] *
             blkv_get_tile_ptr(src1_scale)[idx_1x]);
  }

  size_t idx = index<tile_shape_C>(j, i);
  blkv_get_tile_ptr(dst)[idx] = data;
}

// Matrix Multiply MX:
// C[MxN] = scale(A, AX) x scale(B, BX)
template <is_tile_data_v tile_shape_A,  is_tile_data_v tile_shape_AX,
          is_tile_data_v tile_shape_B,  is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMULMX_Impl(tile_shape_C &dst,
                   tile_shape_A &src0, tile_shape_AX &src0_scale,
                   tile_shape_B &src1, tile_shape_BX &src1_scale) {
  static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
                "Error! Cube A:Columns != Cube B:Rows");

  size_t M = dst.GetValidRow();
  size_t N = dst.GetValidCol();
  size_t K = src0.GetValidCol();

  if constexpr (std::is_same<typename tile_shape_A::DType,
                             typename tile_shape_B::DType>::value) {
    if constexpr ((!tile_shape_A::isBoxedLayout)  &&
                  (!tile_shape_AX::isBoxedLayout) &&
                  (!tile_shape_B::isBoxedLayout)  &&
                  (!tile_shape_BX::isBoxedLayout) &&
                  (!tile_shape_C::isBoxedLayout)) {
      MatMulMx_Vec_Impl<tile_shape_A, tile_shape_AX,
                        tile_shape_B, tile_shape_BX,
                        tile_shape_C>
      <<<N, M, 1>>>(dst.data(),
                    src0.data(), src0_scale.data(),
                    src1.data(), src1_scale.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cube C:FractalSize != 1024 or not Nz_layout");

      static_assert(tile_shape_A::Loc  != Location::Acc &&
                    tile_shape_B::Loc  != Location::Acc,
                    "Error! MatmulMx input can not be ACC");

      static_assert(tile_shape_AX::Loc != Location::Acc &&
                    tile_shape_BX::Loc != Location::Acc,
                    "Error! MatmulMx scale input can not be ACC");

      static_assert(tile_shape_C::Loc == Location::Acc,
                    "Error! MatmulMx output only can be ACC");

      blk_matmulmx(
          M, N, K,
          type_traits<typename tile_shape_A::DType>::TypeCode,
          type_traits<typename tile_shape_B::DType>::TypeCode,
          dst.data(),
          src0.data(), src0_scale.data(),
          src1.data(), src1_scale.data());
    } else {
      static_assert((!tile_shape_A::isBoxedLayout) &&
                    (!tile_shape_B::isBoxedLayout),
                    "Storage layout type not supported");
    }
  } else {
    static_assert(std::is_same<typename tile_shape_A::DType,
                               typename tile_shape_B::DType>::value,
                  "Data type not supported");
  }
}


template <typename tile_shape_A,
          typename tile_shape_B, typename tile_shape_BX,
          typename tile_shape_C>
void __vec__ MatMulMxb_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_B::TileDType __in__ src1,
    const typename tile_shape_BX::TileDType __in__ src1_scale) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  typename tile_shape_C::DType data = 0;

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < tile_shape_A::ValidCol; ++k) {
    size_t idx_0  = index<tile_shape_A>(j, k);
    size_t idx_1  = index<tile_shape_B>(k, i);
    size_t idx_1x = index<tile_shape_BX>(k, i);

    data += blkv_get_tile_ptr(src0)[idx_0] *
            (blkv_get_tile_ptr(src1)[idx_1] *
             blkv_get_tile_ptr(src1_scale)[idx_1x]);
  }

  size_t idx = index<tile_shape_C>(j, i);
  blkv_get_tile_ptr(dst)[idx] = data;
}

// Matrix Multiply MXB:
// C[MxN] = A x scale(B, BX)
template <is_tile_data_v tile_shape_A,
          is_tile_data_v tile_shape_B, is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMULMXB_Impl(tile_shape_C &dst,
                    tile_shape_A &src0,
                    tile_shape_B &src1, tile_shape_BX &src1_scale) {
  static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
                "Error! Cube A:Columns != Cube B:Rows");

  size_t M = dst.GetValidRow();
  size_t N = dst.GetValidCol();
  size_t K = src0.GetValidCol();

  if constexpr (std::is_same<typename tile_shape_A::DType,
                             typename tile_shape_B::DType>::value) {
    if constexpr ((!tile_shape_A::isBoxedLayout)  &&
                  (!tile_shape_B::isBoxedLayout)  &&
                  (!tile_shape_BX::isBoxedLayout) &&
                  (!tile_shape_C::isBoxedLayout)) {
      MatMulMxb_Vec_Impl<tile_shape_A,
                         tile_shape_B, tile_shape_BX,
                         tile_shape_C>
      <<<N, M, 1>>>(dst.data(),
                    src0.data(),
                    src1.data(), src1_scale.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cube C:FractalSize != 1024 or not Nz_layout");

      static_assert(tile_shape_A::Loc != Location::Acc &&
                    tile_shape_B::Loc != Location::Acc,
                    "Error! MatmulMxb input can not be ACC");

      static_assert(tile_shape_BX::Loc != Location::Acc,
                    "Error! MatmulMxb scale input can not be ACC");

      static_assert(tile_shape_C::Loc == Location::Acc,
                    "Error! MatmulMxb output only can be ACC");

      blk_matmulmxb(
          M, N, K,
          type_traits<typename tile_shape_A::DType>::TypeCode,
          type_traits<typename tile_shape_B::DType>::TypeCode,
          dst.data(),
          src0.data(),
          src1.data(), src1_scale.data());
    } else {
      static_assert((!tile_shape_A::isBoxedLayout) &&
                    (!tile_shape_B::isBoxedLayout),
                    "Storage layout type not supported");
    }
  } else {
    static_assert(std::is_same<typename tile_shape_A::DType,
                               typename tile_shape_B::DType>::value,
                  "Data type not supported");
  }
}

#endif

#endif
