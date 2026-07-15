#ifndef MATMACC_HPP
#define MATMACC_HPP

#include "common/pto_tile.hpp"

using namespace pto;

#ifdef __linx
template <typename...>
struct linx_matmacc_unsupported {
  static constexpr bool value = false;
};

// Matrix Multiply and Accumulate: C[MxN] += A[MxK] x B[KxN]
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_C>
void MATMACC_Impl(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
  static_assert(tile_shape_A::ValidCol == tile_shape_B::ValidRow,
                "Linx scalar MATMACC requires A columns to match B rows");
  static_assert(!tile_shape_A::isBoxedLayout && !tile_shape_B::isBoxedLayout &&
                    !tile_shape_C::isBoxedLayout,
                "Linx scalar MATMACC supports only unboxed layouts");
  static_assert(tile_shape_A::Loc != Location::Acc &&
                    tile_shape_B::Loc != Location::Acc &&
                    tile_shape_C::Loc != Location::Acc,
                "Linx scalar MATMACC does not support ACC tile operands");
  static_assert(std::is_integral<typename tile_shape_A::DType>::value &&
                    std::is_integral<typename tile_shape_B::DType>::value &&
                    std::is_integral<typename tile_shape_C::DType>::value,
                "Linx scalar MATMACC direct smoke supports integral tiles only");

  constexpr size_t rows = tile_shape_C::ValidRow;
  constexpr size_t cols = tile_shape_C::ValidCol;
  constexpr size_t inner = tile_shape_A::ValidCol;

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      typename tile_shape_C::DType acc = dst.data()[index<tile_shape_C>(row, col)];
      for (size_t k = 0; k < inner; ++k) {
        acc += src0.data()[index<tile_shape_A>(row, k)] *
               src1.data()[index<tile_shape_B>(k, col)];
      }
      dst.data()[index<tile_shape_C>(row, col)] = acc;
    }
  }
}

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_AX,
          is_tile_data_v tile_shape_B, is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMACCMX_Impl(tile_shape_C &, tile_shape_A &, tile_shape_AX &,
                    tile_shape_B &, tile_shape_BX &) {
  static_assert(linx_matmacc_unsupported<tile_shape_A, tile_shape_AX,
                                         tile_shape_B, tile_shape_BX,
                                         tile_shape_C>::value,
                "Linx direct MATMACCMX smoke is not implemented");
}

template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_BX, is_tile_data_v tile_shape_C>
void MATMACCMXB_Impl(tile_shape_C &, tile_shape_A &, tile_shape_B &,
                     tile_shape_BX &) {
  static_assert(linx_matmacc_unsupported<tile_shape_A, tile_shape_B,
                                         tile_shape_BX, tile_shape_C>::value,
                "Linx direct MATMACCMXB smoke is not implemented");
}

#else

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void __vec__ MatMacc_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_B::TileDType __in__ src1,
    const typename tile_shape_C::TileDType __in__ acc) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  typename tile_shape_C::DType data =
    blkv_get_tile_ptr(acc)[index<tile_shape_C>(j, i)];

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

// Matrix Multiply and Accumulate: C[MxN] += A[M×K] x B[KxN]
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B,
          is_tile_data_v tile_shape_C>
void MATMACC_Impl(tile_shape_C &dst, tile_shape_A &src0, tile_shape_B &src1) {
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
      MatMacc_Vec_Impl<tile_shape_A, tile_shape_B, tile_shape_C>
      <<<N, M, 1>>>(dst.data(), src0.data(), src1.data(), dst.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cude C:FractalSize != 1024 or not Nz_layout");
      static_assert(tile_shape_A::Loc != Location::Acc && tile_shape_B::Loc != Location::Acc, "Error! MatMacc input can not be ACC");
      static_assert(tile_shape_C::Loc == Location::Acc, "Error! MatMacc output only canbe ACC");
      blk_matmul_ac(M, N, K, type_traits<typename tile_shape_A::DType>::TypeCode, type_traits<typename tile_shape_B::DType>::TypeCode,
                    dst.data(), src0.data(), src1.data(), dst.data());
    } else {
      static_assert((!tile_shape_A::isBoxedLayout) &&
                    (!tile_shape_B::isBoxedLayout),
                    "Storage layout type not supported");
    }
}

template <typename tile_shape_A, typename tile_shape_AX,
          typename tile_shape_B, typename tile_shape_BX,
          typename tile_shape_C>
void __vec__ MatMaccMx_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_AX::TileDType __in__ src0x,
    const typename tile_shape_B::TileDType __in__ src1,
    const typename tile_shape_BX::TileDType __in__ src1x,
    const typename tile_shape_C::TileDType __in__ acc) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  typename tile_shape_C::DType data =
      blkv_get_tile_ptr(acc)[index<tile_shape_C>(j, i)];

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < tile_shape_A::ValidCol; ++k) {
    size_t idx_0  = index<tile_shape_A>(j, k);
    size_t idx_0x = index<tile_shape_AX>(j, k);
    size_t idx_1  = index<tile_shape_B>(k, i);
    size_t idx_1x = index<tile_shape_BX>(k, i);

    data += (blkv_get_tile_ptr(src0)[idx_0] * blkv_get_tile_ptr(src0x)[idx_0x]) *
            (blkv_get_tile_ptr(src1)[idx_1] * blkv_get_tile_ptr(src1x)[idx_1x]);
  }

  size_t idx = index<tile_shape_C>(j, i);
  blkv_get_tile_ptr(dst)[idx] = data;
}

// Matrix Multiply MX and Accumulate:
// C[MxN] += scale(A, AX) x scale(B, BX)
template <is_tile_data_v tile_shape_A,  is_tile_data_v tile_shape_AX,
          is_tile_data_v tile_shape_B,  is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMACCMX_Impl(tile_shape_C &dst,
                    tile_shape_A &src0,  tile_shape_AX &src0x,
                    tile_shape_B &src1,  tile_shape_BX &src1x) {
  static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
                "Error! Cude A:Columns != Cude B:Rows");

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
      MatMaccMx_Vec_Impl<tile_shape_A, tile_shape_AX,
                         tile_shape_B, tile_shape_BX,
                         tile_shape_C>
      <<<N, M, 1>>>(dst.data(), src0.data(), src0x.data(),
                    src1.data(), src1x.data(), dst.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cude C:FractalSize != 1024 or not Nz_layout");

      static_assert(tile_shape_A::Loc  != Location::Acc &&
                    tile_shape_AX::Loc != Location::Acc &&
                    tile_shape_B::Loc  != Location::Acc &&
                    tile_shape_BX::Loc != Location::Acc,
                    "Error! MatMaccMx input can not be ACC");

      static_assert(tile_shape_C::Loc == Location::Acc,
                    "Error! MatMaccMx output only canbe ACC");

      blk_matmulmx_ac( M, N, K, type_traits<typename tile_shape_A::DType>::TypeCode,
          type_traits<typename tile_shape_B::DType>::TypeCode, dst.data(),
          src0.data(), src0x.data(), src1.data(), src1x.data(), dst.data());
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
void __vec__ MatMaccMxb_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_B::TileDType __in__ src1,
    const typename tile_shape_BX::TileDType __in__ src1x,
    const typename tile_shape_C::TileDType __in__ acc) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();

  typename tile_shape_C::DType data =
      blkv_get_tile_ptr(acc)[index<tile_shape_C>(j, i)];

  #pragma clang loop unroll(full)
  for (size_t k = 0; k < tile_shape_A::ValidCol; ++k) {
    size_t idx_0  = index<tile_shape_A>(j, k);
    size_t idx_1  = index<tile_shape_B>(k, i);
    size_t idx_1x = index<tile_shape_BX>(k, i);


    data += blkv_get_tile_ptr(src0)[idx_0] *
            (blkv_get_tile_ptr(src1)[idx_1] * blkv_get_tile_ptr(src1x)[idx_1x]);
  }

  size_t idx = index<tile_shape_C>(j, i);
  blkv_get_tile_ptr(dst)[idx] = data;
}

// Matrix Multiply MXB and Accumulate:
// C[MxN] += A x scale(B, BX)
template <is_tile_data_v tile_shape_A,
          is_tile_data_v tile_shape_B,  is_tile_data_v tile_shape_BX,
          is_tile_data_v tile_shape_C>
void MATMACCMXB_Impl(tile_shape_C &dst,
                     tile_shape_A &src0,
                     tile_shape_B &src1, tile_shape_BX &src1x) {
  static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
                "Error! Cude A:Columns != Cude B:Rows");

  size_t M = dst.GetValidRow();
  size_t N = dst.GetValidCol();
  size_t K = src0.GetValidCol();

  if constexpr (std::is_same<typename tile_shape_A::DType,
                             typename tile_shape_B::DType>::value) {
    if constexpr ((!tile_shape_A::isBoxedLayout)  &&
                  (!tile_shape_B::isBoxedLayout)  &&
                  (!tile_shape_BX::isBoxedLayout) &&
                  (!tile_shape_C::isBoxedLayout)) {
      MatMaccMxb_Vec_Impl<tile_shape_A,
                          tile_shape_B, tile_shape_BX,
                          tile_shape_C>
      <<<N, M, 1>>>(dst.data(), src0.data(),
                    src1.data(), src1x.data(),
                    dst.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cude C:FractalSize != 1024 or not Nz_layout");

      static_assert(tile_shape_A::Loc  != Location::Acc &&
                    tile_shape_B::Loc  != Location::Acc &&
                    tile_shape_BX::Loc != Location::Acc,
                    "Error! MatMaccMxb input can not be ACC");

      static_assert(tile_shape_C::Loc == Location::Acc,
                    "Error! MatMaccMxb output only canbe ACC");

      blk_matmulmxb_ac( M, N, K, type_traits<typename tile_shape_A::DType>::TypeCode,
          type_traits<typename tile_shape_B::DType>::TypeCode, dst.data(),
          src0.data(), src1.data(), src1x.data(), dst.data());
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
