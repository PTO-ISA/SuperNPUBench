#ifndef MATMACC_HPP
#define MATMACC_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void __vec__ MatMacc_Vec_Impl(
    typename tile_shape_C::TileDType __out__ dst,
    const typename tile_shape_A::TileDType __in__ src0,
    const typename tile_shape_B::TileDType __in__ src1) {
  size_t i = blkv_get_index_x();
  size_t j = blkv_get_index_y();
  typename tile_shape_C::DType data = 
    blkv_get_tile_ptr(dst)[index<tile_shape_C>(j, i)];

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
  static_assert(tile_shape_A::Cols == tile_shape_B::Rows,
                "Error! Cude A:Columns != Cude B:Rows");

  static constexpr size_t M = tile_shape_C::ValidRow;
  static constexpr size_t N = tile_shape_C::ValidCol;
  static constexpr size_t K = tile_shape_A::ValidCol;
  if constexpr (std::is_same<typename tile_shape_A::DType, 
                             typename tile_shape_B::DType>::value) {
    if constexpr ((!tile_shape_A::isBoxedLayout) && 
                  (!tile_shape_B::isBoxedLayout) &&
                  (!tile_shape_C::isBoxedLayout)) {
      MatMacc_Vec_Impl<tile_shape_A, tile_shape_B, tile_shape_C>
      <<<N, M, 1>>>(dst.data(), src0.data(), src1.data());
    } else if constexpr (is_Nz_layout<tile_shape_A>::value &&
                         is_Zn_layout<tile_shape_B>::value) {
      static_assert(tile_shape_C::SFractalSize == 1024 &&
                        is_Nz_layout<tile_shape_C>::value,
                    "Error! Cude C:FractalSize != 1024 or not Nz_layout");
      blk_matmul_ac(M, N, K, type_traits<typename tile_shape_A::DType>::TypeCode,
                    dst.data(), src0.data(), src1.data(), dst.data());
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