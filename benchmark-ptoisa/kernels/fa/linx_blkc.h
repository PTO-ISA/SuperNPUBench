#ifndef PTO_FA_LINX_BLKC_H
#define PTO_FA_LINX_BLKC_H

#include <common/block_vector_compat.hpp>

#ifndef __bf16
#define __bf16 pto_bf16_t
#endif

using pto::blkv::blkv_fexp;
using pto::blkv::blkv_fsqrt;
using pto::blkv::blkv_get_index_x;
using pto::blkv::blkv_get_index_y;
using pto::blkv::blkv_get_index_z;
using pto::blkv::blkv_get_tile_ptr;
using pto::blkv::blkv_max;

using __fp4_e1m2x2 = pto::fp4_e2m1_t;
using __bf16x2 = uint32_t;

template <typename To, typename From>
inline To linx_cvt_as(From value) {
  return static_cast<To>(value);
}

template <typename To, typename From>
inline void linx_cvt(To &dst, From value) {
  dst = linx_cvt_as<To>(value);
}

template <typename To, typename A, typename B>
inline void linx_cvt_package(To &dst, A a, B b) {
  const uint16_t hi =
      static_cast<uint16_t>(pto::lowp_word_from_bf16(pto::bf16_t{static_cast<float>(a)}) &
                            0xffffu);
  const uint16_t lo =
      static_cast<uint16_t>(pto::lowp_word_from_bf16(pto::bf16_t{static_cast<float>(b)}) &
                            0xffffu);
  if constexpr (sizeof(To) >= 4) {
    dst = static_cast<To>((static_cast<uint32_t>(hi) << 16) | lo);
  } else {
    dst = static_cast<To>(pto::float_to_fp4_e2m1(static_cast<float>(a)));
  }
}

inline void blkv_bf16_fmax(__bf16 &dst, __bf16 a, __bf16 b) {
  dst = a < b ? b : a;
}

inline __bf16 blkv_bf16_max(__bf16 a, __bf16 b) { return a < b ? b : a; }
inline __bf16 blkv_bf16_mul(__bf16 a, __bf16 b) { return a * b; }
inline __bf16 blkv_bf16_div(__bf16 a, __bf16 b) { return a / b; }

inline void blkv_bf16_fadd(__bf16 &dst, __bf16 a, __bf16 b) { dst = a + b; }
inline void blkv_bf16_fsub(__bf16 &dst, __bf16 a, __bf16 b) { dst = a - b; }
inline void blkv_bf16_fmul(__bf16 &dst, __bf16 a, __bf16 b) { dst = a * b; }
inline void blkv_bf16_fdiv(__bf16 &dst, __bf16 a, __bf16 b) { dst = a / b; }
inline void blkv_bf16_fexp(__bf16 &dst, __bf16 a) {
  dst = pto::blkv::blkv_fexp(static_cast<float>(a));
}

inline float linx_bf16x2_hi(__bf16x2 value) {
  return static_cast<float>(pto::bf16_from_lowp_word((value >> 16) & 0xffffu));
}

inline float linx_bf16x2_lo(__bf16x2 value) {
  return static_cast<float>(pto::bf16_from_lowp_word(value & 0xffffu));
}

inline __bf16x2 linx_pack_bf16x2(float hi, float lo) {
  return (pto::lowp_word_from_bf16(pto::bf16_t{hi}) << 16) |
         pto::lowp_word_from_bf16(pto::bf16_t{lo});
}

inline void blkv_bf16x2_fmax(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = linx_pack_bf16x2(pto::blkv::blkv_max(linx_bf16x2_hi(a), linx_bf16x2_hi(b)),
                         pto::blkv::blkv_max(linx_bf16x2_lo(a), linx_bf16x2_lo(b)));
}

inline __bf16x2 blkv_bf16x2_mul(__bf16x2 a, __bf16x2 b) {
  return linx_pack_bf16x2(linx_bf16x2_hi(a) * linx_bf16x2_hi(b),
                         linx_bf16x2_lo(a) * linx_bf16x2_lo(b));
}

inline void blkv_bf16x2_mul(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = blkv_bf16x2_mul(a, b);
}

inline void blkv_bf16x2_fadd(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = linx_pack_bf16x2(linx_bf16x2_hi(a) + linx_bf16x2_hi(b),
                         linx_bf16x2_lo(a) + linx_bf16x2_lo(b));
}

inline void blkv_bf16x2_fsub(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = linx_pack_bf16x2(linx_bf16x2_hi(a) - linx_bf16x2_hi(b),
                         linx_bf16x2_lo(a) - linx_bf16x2_lo(b));
}

inline void blkv_bf16x2_fmul(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = blkv_bf16x2_mul(a, b);
}

inline void blkv_bf16x2_fdiv(__bf16x2 &dst, __bf16x2 a, __bf16x2 b) {
  dst = linx_pack_bf16x2(linx_bf16x2_hi(a) / linx_bf16x2_hi(b),
                         linx_bf16x2_lo(a) / linx_bf16x2_lo(b));
}

inline void blkv_bf16x2_fmsub(__bf16x2 &dst, __bf16x2 a, __bf16x2 b,
                              __bf16x2 c) {
  dst = linx_pack_bf16x2((linx_bf16x2_hi(a) * linx_bf16x2_hi(b)) -
                             linx_bf16x2_hi(c),
                         (linx_bf16x2_lo(a) * linx_bf16x2_lo(b)) -
                             linx_bf16x2_lo(c));
}

inline void blkv_bf16x2_fexp(__bf16x2 &dst, __bf16x2 a) {
  dst = linx_pack_bf16x2(pto::blkv::blkv_fexp(linx_bf16x2_hi(a)),
                         pto::blkv::blkv_fexp(linx_bf16x2_lo(a)));
}

#endif // PTO_FA_LINX_BLKC_H
