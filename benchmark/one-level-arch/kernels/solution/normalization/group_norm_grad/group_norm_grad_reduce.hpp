// =============================================================================
// group_norm_grad_reduce.hpp — single-pass reduce helpers shared by the
// GroupNorm backward kernels (HxW > 1 and HxW == 1).
//
// A reduction over axis R finishes in one pass with Tiles of at most 32 KiB
// (FP32 CubeM32 [32,256]):
//   R = 2^k <= 256     : up to 32 rows at once, [rows,R] -> TROWSUM -> [rows,1].
//   R = fa*fr <= 8192  : one row folded to [fa,fr] (fa <= 32, fr = 2^k <= 256)
//                        -> TROWSUM -> TCOLSUM -> scalar (as rms_norm).
// CUBE row reductions require the source valid width to equal the physical
// Tile width, and CUBE TLOAD leaves columns past the valid width undefined, so
// Tile columns are always exactly R (rows form) or fr (fold form).
// =============================================================================
#ifndef SUPERNPU_GROUP_NORM_GRAD_REDUCE_PTO_HPP
#define SUPERNPU_GROUP_NORM_GRAD_REDUCE_PTO_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace gn_grad_reduce {

using GFloat = global_tensor<float, RowMajor<-1, -1>>;

// Tile column counts with a [32,Cols] instantiation.
inline bool is_tile_cols(int64_t cols) {
  return cols >= 2 && cols <= 256 && (cols & (cols - 1)) == 0;
}

// Smallest fr with R = fa*fr, fa <= 32 and is_tile_cols(fr); 0 if none.
inline int64_t fold_cols(int64_t R) {
  for (int64_t fr = 2; fr <= 256; fr *= 2)
    if (fr * 32 >= R && R % fr == 0)
      return fr;
  return 0;
}

// Run f<Cols>() with Cols == cols; cols must satisfy is_tile_cols().
template <typename F> inline void with_cols(int64_t cols, F &&f) {
  switch (cols) {
  case 2:
    f.template operator()<2>();
    break;
  case 4:
    f.template operator()<4>();
    break;
  case 8:
    f.template operator()<8>();
    break;
  case 16:
    f.template operator()<16>();
    break;
  case 32:
    f.template operator()<32>();
    break;
  case 64:
    f.template operator()<64>();
    break;
  case 128:
    f.template operator()<128>();
    break;
  default:
    f.template operator()<256>();
    break;
  }
}

template <typename dtype, int Cols> struct Tiles {
  using H = Tile<Location::Vec, dtype, 32, Cols, BLayout::CubeM32, -1, -1>;
  using F = Tile<Location::Vec, float, 32, Cols, BLayout::CubeM32, -1, -1>;
  // PTO #311: a row-reduction dest keeps the source physical columns.
  using W = Tile<Location::Vec, float, 32, Cols, BLayout::CubeM32, -1, 1>;
  using S = Tile<Location::Vec, float, 32, Cols, BLayout::CubeM32, 1, 1>;
  using V = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, -1, 1>;
  using Scalar = Tile<Location::Vec, float, 32, 1, BLayout::CubeM32, 1, 1>;
};

// c2 = (sum2*mean - sum1) * rstd^3 * s;  c3 = -c2*mean - sum2*rstd*s.
// sum1 is clobbered.
template <typename T>
__attribute__((always_inline)) inline void
c2_c3(T &c2, T &c3, T &sum1, T &sum2, T &mean, T &rstd, float s) {
  TMUL(c2, sum2, mean);
  TSUB(c2, c2, sum1);
  TMUL(c3, rstd, rstd);
  TMUL(c3, c3, rstd);
  TMUL(c2, c2, c3);
  TMULS(c2, c2, s);
  TMUL(c3, c2, mean);
  TMULS(c3, c3, -1.0f);
  TMUL(sum1, sum2, rstd);
  TMULS(sum1, sum1, s);
  TSUB(c3, c3, sum1);
}

// Per-row sums of [rows,R] Tiles w1/w2 -> c2/c3 for those rows.
template <typename T>
__attribute__((always_inline)) inline void
c2_c3_rows(typename T::W &w1, typename T::W &w2, GFloat &gmean, GFloat &grstd,
           GFloat &gc2, GFloat &gc3, int64_t rows, float s) {
  using V = typename T::V;
  V zero(rows), sum1(rows), sum2(rows), m(rows), r(rows), c2(rows), c3(rows);
  auto v1 = TREDUCEPREFIXVIEW<V>(w1);
  auto v2 = TREDUCEPREFIXVIEW<V>(w2);
  TEXPANDS(zero, 0.0f);
  TADD(sum1, zero, v1);
  TADD(sum2, zero, v2);
  TLOAD(m, gmean);
  TLOAD(r, grstd);
  c2_c3(c2, c3, sum1, sum2, m, r, s);
  TSTORE(gc2, c2);
  TSTORE(gc3, c3);
}

// Folded [fa,fr] row sums w1/w2 -> one (c2, c3) pair.
template <typename T>
__attribute__((always_inline)) inline void
c2_c3_fold(typename T::W &w1, typename T::W &w2, GFloat &gmean, GFloat &grstd,
           GFloat &gc2, GFloat &gc3, float s) {
  using Sc = typename T::Scalar;
  typename T::S s1, s2;
  TCOLSUM(s1, w1);
  TCOLSUM(s2, w2);
  Sc sum1, sum2, m, r, c2, c3;
  auto v1 = TREDUCEPREFIXVIEW<Sc>(s1);
  auto v2 = TREDUCEPREFIXVIEW<Sc>(s2);
  TMULS(sum1, v1, 1.0f);
  TMULS(sum2, v2, 1.0f);
  TLOAD(m, gmean);
  TLOAD(r, grstd);
  c2_c3(c2, c3, sum1, sum2, m, r, s);
  TSTORE(gc2, c2);
  TSTORE(gc3, c3);
}

} // namespace gn_grad_reduce

#endif // SUPERNPU_GROUP_NORM_GRAD_REDUCE_PTO_HPP
