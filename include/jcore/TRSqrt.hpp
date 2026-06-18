#ifndef TRSQRT_HPP
#define TRSQRT_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void __vec__
TRSqrt_RowMajor_Imp_f16(typename tile_shape::TileDType __out__ dst,
                        const typename tile_shape::TileDType __in__ src,
                        const uint16_t __in__ col) {
  __asm__ __volatile__(
      "l.madd   lc1.uh, a0.sh, lc0.uh,  ->vu.h\n" // index = i * col + j
      "l.lh     [TA, vu#1.sh<<1],  ->vt.h\n"      // src[index]
      "l.fsqrt  vt#1.fh,  ->vt.h\n"               // sqrt(src)
      "l.frecip vt#1.fh,  ->vt.h\n"               // recip(src)
      "l.sh     vt#1.sh, [TO, vu#1.sh<<1]\n"      // dst[index] = sqrt(src)
      "c.bstop\n"
      :
      :
      : "memory");
}

template <typename tile_shape>
void __vec__
TRSqrt_RowMajor_Imp_f32(typename tile_shape::TileDType __out__ dst,
                        const typename tile_shape::TileDType __in__ src,
                        const uint16_t __in__ col) {
  __asm__ __volatile__(
      "l.madd   lc1.uh, a0.sh, lc0.uh,  ->vu.h\n" // index = i * col + j
      "l.lw     [TA, vu#1.sh<<2],  ->vt.w\n"      // src[index]
      "l.fsqrt  vt#1.fs,  ->vt.w\n"               // sqrt(src)
      "l.frecip vt#1.fs,  ->vt.w\n"               // recip(src)
      "l.sw     vt#1.sw, [TO, vu#1.sh<<2]\n"      // dst[index] = sqrt(src)
      "c.bstop\n"
      :
      :
      : "memory");
}

template <is_tile_data_v tile_shape>
void TRSQRT_Impl(tile_shape &dst, tile_shape &src) {
  static constexpr uint16_t row = tile_shape::ValidRow;
  static constexpr uint16_t col = tile_shape::ValidCol;
  if constexpr (tile_shape::isRowMajor) {
    if constexpr (std::is_same<typename tile_shape::DType, float>::value) {
      TRSqrt_RowMajor_Imp_f32<tile_shape>
          <<<col, row, 1>>>(dst.data(), src.data(), tile_shape::RowStride);
    } else {
      static_assert(std::is_same<typename tile_shape::DType, float>::value,
                    "Data type not supported");
    }
  } else {
    static_assert(tile_shape::isRowMajor,
                  "Storage layout type not supported");
  }
}

#endif