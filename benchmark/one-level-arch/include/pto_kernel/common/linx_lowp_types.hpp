#ifndef PTO_COMMON_LINX_LOWP_TYPES_HPP
#define PTO_COMMON_LINX_LOWP_TYPES_HPP

#include <stdint.h>

namespace pto {

struct fp16_t {
  uint16_t bits;
};

#if !defined(__CPU_SIM)
using half = fp16_t;
#endif

struct fp8_e4m3_t {
  uint8_t bits;
};

struct fp4_e2m1_t {
  uint8_t bits;
};

inline float fp16_to_float(fp16_t x) {
  const uint16_t v = x.bits;
  const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
  const uint32_t exp = (v >> 10) & 0x1fu;
  const uint32_t mant = v & 0x03ffu;

  uint32_t out_bits = 0u;
  if (exp == 0u) {
    if (mant == 0u) {
      out_bits = sign;
    } else {
      int e = -14;
      uint32_t m = mant;
      while ((m & 0x0400u) == 0u) {
        m <<= 1u;
        --e;
      }
      m &= 0x03ffu;
      const uint32_t exp32 = static_cast<uint32_t>(e + 127);
      out_bits = sign | (exp32 << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    out_bits = sign | 0x7f800000u | (mant << 13);
  } else {
    const uint32_t exp32 = exp + (127u - 15u);
    out_bits = sign | (exp32 << 23) | (mant << 13);
  }

  union {
    uint32_t u;
    float f;
  } cvt = {out_bits};
  return cvt.f;
}

inline fp16_t float_to_fp16(float x) {
  union {
    float f;
    uint32_t u;
  } cvt = {x};

  const uint32_t sign = (cvt.u >> 16) & 0x8000u;
  const int exp32 = static_cast<int>((cvt.u >> 23) & 0xffu);
  const uint32_t mant32 = cvt.u & 0x7fffffu;

  if (exp32 == 0xff) {
    const uint16_t nan_inf = static_cast<uint16_t>(sign | 0x7c00u | (mant32 ? 0x0200u : 0u));
    return fp16_t{nan_inf};
  }

  const int exp16 = exp32 - 127 + 15;
  if (exp16 <= 0) {
    if (exp16 < -10)
      return fp16_t{static_cast<uint16_t>(sign)};
    uint32_t mant = mant32 | 0x800000u;
    const int shift = 14 - exp16;
    uint32_t rounded = mant >> static_cast<uint32_t>(shift);
    if (((mant >> static_cast<uint32_t>(shift - 1)) & 1u) != 0u)
      ++rounded;
    return fp16_t{static_cast<uint16_t>(sign | (rounded & 0x03ffu))};
  }

  if (exp16 >= 31)
    return fp16_t{static_cast<uint16_t>(sign | 0x7c00u)};

  uint32_t mant = mant32;
  mant += 0x1000u; // round-to-nearest-even at fp16 mantissa boundary
  if (mant & 0x800000u) {
    mant = 0u;
    if (exp16 + 1 >= 31)
      return fp16_t{static_cast<uint16_t>(sign | 0x7c00u)};
    return fp16_t{static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp16 + 1) << 10))};
  }

  return fp16_t{static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp16) << 10) | (mant >> 13))};
}

inline float fp8_e4m3_to_float(fp8_e4m3_t x) {
  auto pow2i = [](int e) -> float {
    float s = 1.0f;
    if (e >= 0) {
      for (int i = 0; i < e; ++i)
        s *= 2.0f;
    } else {
      for (int i = 0; i < -e; ++i)
        s *= 0.5f;
    }
    return s;
  };
  const uint8_t bits = x.bits;
  const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
  const uint8_t exp = static_cast<uint8_t>((bits >> 3) & 0x0fu);
  const uint8_t mant = static_cast<uint8_t>(bits & 0x07u);

  if (exp == 0u) {
    if (mant == 0u)
      return 0.0f * sign;
    return sign * (static_cast<float>(mant) / 8.0f) * pow2i(-6);
  }
  if (exp == 0x0fu) {
    const float sat = (1.0f + (7.0f / 8.0f)) * pow2i(7);
    return sign * sat;
  }
  return sign * (1.0f + static_cast<float>(mant) / 8.0f) *
         pow2i(static_cast<int>(exp) - 7);
}

inline fp8_e4m3_t float_to_fp8_e4m3(float x) {
  if (x == 0.0f)
    return fp8_e4m3_t{0u};

  const bool neg = x < 0.0f;
  float ax = neg ? -x : x;
  int e = 0;
  float norm = ax;
  while (norm >= 2.0f && e < 30) {
    norm *= 0.5f;
    ++e;
  }
  while (norm < 1.0f && e > -30) {
    norm *= 2.0f;
    --e;
  }
  int ef = e + 7;

  uint8_t sign = neg ? 0x80u : 0u;

  if (ef <= 0) {
    int mant = static_cast<int>(ax * 512.0f + 0.5f); // ax * 2^(6+3)
    if (mant < 0)
      mant = 0;
    if (mant > 7)
      mant = 7;
    return fp8_e4m3_t{static_cast<uint8_t>(sign | mant)};
  }

  if (ef >= 0x0f)
    return fp8_e4m3_t{static_cast<uint8_t>(sign | 0x7eu)};

  float frac = norm - 1.0f;
  int mant = static_cast<int>(frac * 8.0f + 0.5f);
  if (mant >= 8) {
    mant = 0;
    ++ef;
    if (ef >= 0x0f)
      return fp8_e4m3_t{static_cast<uint8_t>(sign | 0x7eu)};
  }
  if (mant < 0)
    mant = 0;
  return fp8_e4m3_t{static_cast<uint8_t>(sign | (static_cast<uint8_t>(ef) << 3) | static_cast<uint8_t>(mant))};
}

inline float fp4_e2m1_to_float(fp4_e2m1_t x) {
  const uint8_t b = static_cast<uint8_t>(x.bits & 0x0fu);
  static constexpr float kTable[16] = {
      0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kTable[b];
}

inline fp4_e2m1_t float_to_fp4_e2m1(float x) {
  static constexpr float kTable[16] = {
      0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

  auto absf = [](float v) -> float { return v < 0.0f ? -v : v; };
  uint8_t best = 0u;
  float best_err = absf(x - kTable[0]);
  for (uint8_t i = 1u; i < 16u; ++i) {
    const float err = absf(x - kTable[i]);
    if (err < best_err) {
      best_err = err;
      best = i;
    }
  }
  return fp4_e2m1_t{static_cast<uint8_t>(best & 0x0fu)};
}

inline uint32_t lowp_word_from_fp16(fp16_t x) { return static_cast<uint32_t>(x.bits); }
inline uint32_t lowp_word_from_fp8(fp8_e4m3_t x) { return static_cast<uint32_t>(x.bits); }
inline uint32_t lowp_word_from_fp4(fp4_e2m1_t x) { return static_cast<uint32_t>(x.bits & 0x0fu); }

inline fp16_t fp16_from_lowp_word(uint32_t word) { return fp16_t{static_cast<uint16_t>(word & 0xffffu)}; }
inline fp8_e4m3_t fp8_from_lowp_word(uint32_t word) { return fp8_e4m3_t{static_cast<uint8_t>(word & 0xffu)}; }
inline fp4_e2m1_t fp4_from_lowp_word(uint32_t word) { return fp4_e2m1_t{static_cast<uint8_t>(word & 0x0fu)}; }

} // namespace pto

#endif // PTO_COMMON_LINX_LOWP_TYPES_HPP
