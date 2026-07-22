#ifndef PTO_LINX_IMPL_BACKEND_HPP
#define PTO_LINX_IMPL_BACKEND_HPP

#include <stdint.h>
#if defined(PTO_HOST_SIM)
#include <math.h>
#include <string.h>
#endif
#include <pto_kernel/common/linx_lowp_types.hpp>

namespace pto {
namespace linx {
namespace detail {

template <typename... Ts> struct dependent_false {
  static constexpr bool value = false;
};

template <typename A, typename B> struct is_same {
  static constexpr bool value = false;
};

template <typename T> struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T> struct is_arithmetic {
  static constexpr bool value = false;
};

template <> struct is_arithmetic<int> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<unsigned> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<long long> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<unsigned long long> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<float> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<double> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<pto::fp16_t> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<pto::fp8_e4m3_t> {
  static constexpr bool value = true;
};

template <> struct is_arithmetic<pto::fp4_e2m1_t> {
  static constexpr bool value = true;
};

template <typename T> struct is_floating_point {
  static constexpr bool value = false;
};

template <> struct is_floating_point<float> {
  static constexpr bool value = true;
};

template <> struct is_floating_point<double> {
  static constexpr bool value = true;
};

template <typename T> struct DTypeCode {
  static_assert(dependent_false<T>::value,
                "PTO Linx canonical v0.57: unsupported tile dtype");
};

template <> struct DTypeCode<int> {
  static constexpr unsigned value = 17u;
};

template <> struct DTypeCode<unsigned> {
  static constexpr unsigned value = 25u;
};

template <> struct DTypeCode<float> {
  static constexpr unsigned value = 1u;
};

template <> struct DTypeCode<signed char> {
  static constexpr unsigned value = 19u;
};

template <> struct DTypeCode<unsigned char> {
  static constexpr unsigned value = 27u;
};

template <> struct DTypeCode<short> {
  static constexpr unsigned value = 18u;
};

template <> struct DTypeCode<unsigned short> {
  static constexpr unsigned value = 26u;
};

template <> struct DTypeCode<long long> {
  static constexpr unsigned value = 16u;
};

template <> struct DTypeCode<unsigned long long> {
  static constexpr unsigned value = 24u;
};

template <> struct DTypeCode<double> {
  static constexpr unsigned value = 0u;
};

template <> struct DTypeCode<pto::fp16_t> {
  static constexpr unsigned value = 2u;
};

template <> struct DTypeCode<pto::fp8_e4m3_t> {
  static constexpr unsigned value = 3u;
};

template <> struct DTypeCode<pto::fp4_e2m1_t> {
  static constexpr unsigned value = 11u;
};

constexpr unsigned kMinTileBytes = 512u;
constexpr unsigned kMaxTileBytes = 4096u;
constexpr unsigned kTileWords = kMaxTileBytes / sizeof(uint32_t);

#if defined(PTO_HOST_SIM)
struct RawTile {
  alignas(64) uint32_t words[kTileWords];
};
#else
using RawTile = int __attribute__((__vector_size__(4096), __aligned__(64)));
#endif

constexpr unsigned clampTileBytes(unsigned bytes) {
  return bytes < kMinTileBytes
             ? kMinTileBytes
             : (bytes > kMaxTileBytes ? kMaxTileBytes : bytes);
}

constexpr unsigned nextPow2(unsigned value) {
  unsigned p = 1u;
  while (p < value && p < kMaxTileBytes)
    p <<= 1u;
  return p;
}

constexpr unsigned sizeCodeFromBytes(unsigned bytes) {
  const unsigned clipped = clampTileBytes(bytes);
  const unsigned p2 = nextPow2(clipped);
  unsigned code = 0u;
  while ((1u << (code + 4u)) < p2)
    ++code;
  if (code < 5u)
    code = 5u;
  if (code > 8u)
    code = 8u;
  return code;
}

constexpr unsigned dtypeElemBits(unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 0u:  // FP64
  case 16u: // INT64
  case 24u: // UINT64
    return 64u;
  case 1u:  // FP32
  case 17u: // INT32
  case 25u: // UINT32
    return 32u;
  case 2u:  // FP16
  case 6u:  // BF16
  case 18u: // INT16
  case 26u: // UINT16
    return 16u;
  case 3u:  // FP8
  case 7u:  // FPL8
  case 19u: // INT8
  case 27u: // UINT8
    return 8u;
  case 11u: // FP4
  case 12u: // FPL4
  case 20u: // INT4
  case 28u: // UINT4
    return 4u;
  default:
    return 32u;
  }
}

constexpr unsigned dtypeElemBytesForStorage(unsigned dtype) {
  const unsigned bits = dtypeElemBits(dtype);
  return (bits + 7u) / 8u;
}

constexpr unsigned dtypeElemCountForBytes(uint64_t bytes, unsigned dtype) {
  const unsigned bits = dtypeElemBits(dtype);
  if (bits == 0u)
    return 0u;
  const uint64_t total_bits = bytes * 8u;
  return static_cast<unsigned>(total_bits / bits);
}

template <typename Scalar> inline long long encodeScalar(Scalar value) {
  static_assert(is_arithmetic<Scalar>::value,
                "PTO Linx canonical v0.57: scalar operand must be arithmetic");
  if constexpr (is_same<Scalar, pto::fp16_t>::value) {
    return static_cast<long long>(value.bits);
  } else if constexpr (is_same<Scalar, pto::fp8_e4m3_t>::value) {
    return static_cast<long long>(value.bits);
  } else if constexpr (is_same<Scalar, pto::fp4_e2m1_t>::value) {
    return static_cast<long long>(value.bits & 0x0fu);
  }
  if constexpr (is_floating_point<Scalar>::value) {
    if constexpr (sizeof(Scalar) == sizeof(uint32_t)) {
      union {
        Scalar f;
        uint32_t u;
      } cvt = {value};
      return static_cast<long long>(cvt.u);
    } else if constexpr (sizeof(Scalar) == sizeof(uint64_t)) {
      union {
        Scalar f;
        uint64_t u;
      } cvt = {value};
      return static_cast<long long>(cvt.u);
    } else {
      return static_cast<long long>(value);
    }
  }
  return static_cast<long long>(value);
}

#if defined(PTO_HOST_SIM)

inline uint64_t sizeBytesFromCode(unsigned size_code) {
  return (size_code < 60u) ? (1ull << (size_code + 4u)) : 0ull;
}

template <typename T> inline uint32_t bitCastToU32(T value) {
  static_assert(sizeof(T) == sizeof(uint32_t),
                "bitCastToU32 requires 32-bit type");
  uint32_t out = 0;
  memcpy(&out, &value, sizeof(uint32_t));
  return out;
}

template <typename T> inline T bitCastFromU32(uint32_t bits) {
  static_assert(sizeof(T) == sizeof(uint32_t),
                "bitCastFromU32 requires 32-bit type");
  T out{};
  memcpy(&out, &bits, sizeof(uint32_t));
  return out;
}

inline float scalarAsF32(long long scalar_bits) {
  uint32_t bits = static_cast<uint32_t>(scalar_bits & 0xffffffffull);
  return bitCastFromU32<float>(bits);
}

inline int32_t scalarAsI32(long long scalar_bits) {
  return static_cast<int32_t>(scalar_bits & 0xffffffffull);
}

inline uint32_t scalarToWordDType(long long scalar_bits, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
    return static_cast<uint32_t>(static_cast<uint16_t>(scalar_bits & 0xffffu));
  case 3u:
    return static_cast<uint32_t>(static_cast<uint8_t>(scalar_bits & 0xffu));
  case 11u:
    return static_cast<uint32_t>(static_cast<uint8_t>(scalar_bits & 0x0fu));
  default:
    return static_cast<uint32_t>(scalar_bits & 0xffffffffu);
  }
}

inline uint32_t quantizeF32ToWord(float x, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
    return pto::lowp_word_from_fp16(pto::float_to_fp16(x));
  case 3u:
    return pto::lowp_word_from_fp8(pto::float_to_fp8_e4m3(x));
  case 11u:
    return pto::lowp_word_from_fp4(pto::float_to_fp4_e2m1(x));
  case 17u:
    return static_cast<uint32_t>(static_cast<int32_t>(x));
  case 25u:
    return static_cast<uint32_t>(x < 0.0f ? 0.0f : x);
  case 18u:
    return static_cast<uint16_t>(static_cast<int16_t>(x));
  case 26u:
    return static_cast<uint16_t>(x < 0.0f ? 0.0f : x);
  case 19u:
    return static_cast<uint8_t>(static_cast<int8_t>(
        x < -128.0f ? -128.0f : (x > 127.0f ? 127.0f : x)));
  case 27u:
    return static_cast<uint8_t>(
        x < 0.0f ? 0.0f : (x > 255.0f ? 255.0f : x));
  default:
    return bitCastToU32<float>(x);
  }
}

inline float dequantWordToF32(uint32_t word, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
    return pto::fp16_to_float(pto::fp16_from_lowp_word(word));
  case 3u:
    return pto::fp8_e4m3_to_float(pto::fp8_from_lowp_word(word));
  case 11u:
    return pto::fp4_e2m1_to_float(pto::fp4_from_lowp_word(word));
  case 17u:
    return static_cast<float>(static_cast<int32_t>(word));
  case 25u:
    return static_cast<float>(word);
  case 18u:
    return static_cast<float>(static_cast<int16_t>(word));
  case 26u:
    return static_cast<float>(static_cast<uint16_t>(word));
  case 19u:
    return static_cast<float>(static_cast<int8_t>(word));
  case 27u:
    return static_cast<float>(static_cast<uint8_t>(word));
  default:
    return bitCastFromU32<float>(word);
  }
}

template <unsigned TileOpcode, unsigned DType, unsigned SrcDType = DType>
inline RawTile teplUnaryHost(const RawTile &src, unsigned elems, unsigned rows,
                             unsigned cols) {
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  switch (TileOpcode & 0x3ffu) {
  case 0x00du: // TCVT
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], SrcDType);
      out.words[i] = quantizeF32ToWord(f, DType);
    }
    break;
  case 0x00bu: // TRELU
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(f > 0.0f ? f : 0.0f, DType);
    }
    break;
  case 0x00eu: // TEXP
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(expf(f), DType);
    }
    break;
  case 0x018u: // TRECIP
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      float inv = (f == 0.0f) ? 0.0f : (1.0f / f);
      out.words[i] = quantizeF32ToWord(inv, DType);
    }
    break;
  case 0x00fu: // TLOG
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(f > 0.0f ? logf(f) : -INFINITY, DType);
    }
    break;
  case 0x010u: // TSQRT
  case 0x011u: // TRSQRT
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      const float root = f >= 0.0f ? sqrtf(f) : NAN;
      out.words[i] = quantizeF32ToWord((TileOpcode & 0x3ffu) == 0x011u
                                           ? (root == 0.0f ? 0.0f : 1.0f / root)
                                           : root,
                                       DType);
    }
    break;
  case 0x02du: // TABS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(fabsf(f), DType);
    }
    break;
  case 0x02eu: // TNOT
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = ~src.words[i];
    break;
  case 0x012u:   // TROWMAX
  case 0x013u:   // TROWMIN
  case 0x014u: { // TROWSUM
    for (unsigned r = 0; r < rows; ++r) {
      float value = dequantWordToF32(src.words[r * cols], DType);
      if ((TileOpcode & 0x3ffu) == 0x014u)
        value = 0.0f;
      for (unsigned c = 0; c < cols; ++c) {
        const float cur = dequantWordToF32(src.words[r * cols + c], DType);
        if ((TileOpcode & 0x3ffu) == 0x012u)
          value = value > cur ? value : cur;
        else if ((TileOpcode & 0x3ffu) == 0x013u)
          value = value < cur ? value : cur;
        else
          value += cur;
      }
      out.words[r * cols] = quantizeF32ToWord(value, DType);
    }
    break;
  }
  case 0x015u:   // TCOLMAX
  case 0x016u:   // TCOLMIN
  case 0x017u: { // TCOLSUM
    for (unsigned c = 0; c < cols; ++c) {
      float value = dequantWordToF32(src.words[c], DType);
      if ((TileOpcode & 0x3ffu) == 0x017u)
        value = 0.0f;
      for (unsigned r = 0; r < rows; ++r) {
        const float cur = dequantWordToF32(src.words[r * cols + c], DType);
        if ((TileOpcode & 0x3ffu) == 0x015u)
          value = value > cur ? value : cur;
        else if ((TileOpcode & 0x3ffu) == 0x016u)
          value = value < cur ? value : cur;
        else
          value += cur;
      }
      out.words[c] = quantizeF32ToWord(value, DType);
    }
    break;
  }
  case 0x01du: { // TTRANSPOSE
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned c = 0; c < cols; ++c)
        out.words[c * rows + r] = src.words[r * cols + c];
    break;
  }
  case 0x01eu:   // TCOLEXPAND
  case 0x01fu: { // TROWEXPAND
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned c = 0; c < cols; ++c)
        out.words[r * cols + c] = (TileOpcode & 0x3ffu) == 0x01eu
                                      ? src.words[c]
                                      : src.words[r * cols];
    break;
  }
  case 0x0c0u: { // TSORT
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = src.words[i];
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned i = 1; i < cols; ++i) {
        const uint32_t key = out.words[r * cols + i];
        const float key_f = dequantWordToF32(key, DType);
        unsigned j = i;
        while (j > 0 &&
               dequantWordToF32(out.words[r * cols + j - 1], DType) > key_f) {
          out.words[r * cols + j] = out.words[r * cols + j - 1];
          --j;
        }
        out.words[r * cols + j] = key;
      }
    break;
  }
  case 0x0c2u: // THISTOGRAM
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[src.words[i] % (cols == 0u ? 1u : cols)] += 1u;
    break;
  default:
    // Unsupported op in host backend: keep destination zeroed.
    break;
  }
  return out;
}

template <unsigned TileOpcode, unsigned DType>
inline RawTile teplBinaryHost(const RawTile &lhs, const RawTile &rhs,
                              unsigned elems) {
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  switch (TileOpcode & 0x3ffu) {
  case 0x01au: // TGATHER
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[rhs.words[i] % elems];
    break;
  case 0x01bu: // TSCATTER
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[rhs.words[i] % elems] = lhs.words[i];
    break;
  case 0x000u: // TADD
  case 0x020u: // TADDS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a + b, DType);
    }
    break;
  case 0x001u: // TSUB
  case 0x021u: // TSUBS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a - b, DType);
    }
    break;
  case 0x002u:   // TMUL
  case 0x022u: { // TMULS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a * b, DType);
    }
    break;
  }
  case 0x003u:   // TDIV
  case 0x023u: { // TDIVS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      const float q = (b == 0.0f) ? 0.0f : (a / b);
      out.words[i] = quantizeF32ToWord(q, DType);
    }
    break;
  }
  case 0x004u:   // TMAX
  case 0x024u: { // TMAXS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a > b ? a : b, DType);
    }
    break;
  }
  case 0x005u:   // TMIN
  case 0x025u: { // TMINS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a < b ? a : b, DType);
    }
    break;
  }
  case 0x006u: // TAND
  case 0x026u: // TANDS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] & rhs.words[i];
    break;
  case 0x007u: // TOR
  case 0x027u: // TORS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] | rhs.words[i];
    break;
  case 0x008u: // TXOR
  case 0x028u: // TXORS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] ^ rhs.words[i];
    break;
  case 0x009u: // TSHL
  case 0x029u: // TSHLS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] << (rhs.words[i] & 31u);
    break;
  case 0x00au: // TSHR
  case 0x02au: // TSHRS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] >> (rhs.words[i] & 31u);
    break;
  default:
    break;
  }
  return out;
}

#endif

template <unsigned SizeCode, unsigned DType, long long Layout>
inline RawTile tileTLoad(const void *base, unsigned valid_col,
                         unsigned valid_row, unsigned physical_col,
                         uint64_t stride_bytes) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elem_bits = dtypeElemBits(DType);
  if (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bits == 0u ||
      (bytes64 % elem_bytes) != 0u)
    return out;

  const unsigned max_elems = dtypeElemCountForBytes(bytes64, DType);
  const uint64_t cols = valid_col;
  const uint64_t rows = valid_row;
  const uint64_t physical_cols = physical_col;
  if (rows == 0u || cols == 0u || physical_cols < cols)
    return out;
  if (rows > (UINT64_MAX / physical_cols))
    return out;
  if (rows * physical_cols > max_elems)
    return out;

  const uint64_t row_span_bits = cols * elem_bits;
  const uint64_t row_span_bytes = (row_span_bits + 7u) / 8u;
  stride_bytes = stride_bytes > 0 ? stride_bytes : row_span_bytes;
  if (stride_bytes < row_span_bytes ||
      (elem_bytes != 0u && (stride_bytes % elem_bytes) != 0u)) {
    return out;
  }

  const uint8_t *src = reinterpret_cast<const uint8_t *>(base);
  for (uint64_t r = 0; r < rows; ++r) {
    const uint64_t row_base = r * stride_bytes;
    for (uint64_t c = 0; c < cols; ++c) {
      const uint64_t idx64 = r * physical_cols + c;
      if (idx64 >= kTileWords)
        return out;
      const unsigned idx = static_cast<unsigned>(idx64);

      uint32_t value = 0u;
      if (elem_bits == 4u) {
        const uint64_t byte_addr = row_base + (c >> 1u);
        const uint8_t packed = src[byte_addr];
        value = ((c & 1u) == 0u) ? (packed & 0x0fu) : ((packed >> 4u) & 0x0fu);
      } else if (elem_bytes == 1u) {
        value = static_cast<uint32_t>(src[row_base + c]);
      } else if (elem_bytes == 2u) {
        uint16_t v = 0u;
        memcpy(&v, src + row_base + c * 2u, sizeof(v));
        value = static_cast<uint32_t>(v);
      } else if (elem_bytes == 4u) {
        uint32_t v = 0u;
        memcpy(&v, src + row_base + c * 4u, sizeof(v));
        value = v;
      } else if (elem_bytes == 8u) {
        uint64_t v = 0u;
        memcpy(&v, src + row_base + c * 8u, sizeof(v));
        value = static_cast<uint32_t>(v & 0xffffffffu);
      } else {
        return out;
      }
      out.words[idx] = value;
    }
  }
  return out;
#else
  return __builtin_linx_tile_tload(base, SizeCode, DType, Layout, valid_col,
                                   valid_row, physical_col, stride_bytes);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout>
inline void tileTStore(void *base, RawTile tile, unsigned valid_col,
                       unsigned valid_row, unsigned physical_col,
                       uint64_t stride_bytes) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elem_bits = dtypeElemBits(DType);
  if (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bits == 0u ||
      (bytes64 % elem_bytes) != 0u)
    return;

  const unsigned max_elems = dtypeElemCountForBytes(bytes64, DType);
  const uint64_t cols = valid_col;
  const uint64_t rows = valid_row;
  const uint64_t physical_cols = physical_col;
  if (rows == 0u || cols == 0u || physical_cols < cols)
    return;
  if (rows > (UINT64_MAX / physical_cols))
    return;
  if (rows * physical_cols > max_elems)
    return;

  const uint64_t row_span_bits = cols * elem_bits;
  const uint64_t row_span_bytes = (row_span_bits + 7u) / 8u;
  stride_bytes = stride_bytes > 0 ? stride_bytes : row_span_bytes;
  if (stride_bytes < row_span_bytes ||
      (elem_bytes != 0u && (stride_bytes % elem_bytes) != 0u)) {
    return;
  }

  uint8_t *dst = reinterpret_cast<uint8_t *>(base);
  for (uint64_t r = 0; r < rows; ++r) {
    const uint64_t row_base = r * stride_bytes;
    for (uint64_t c = 0; c < cols; ++c) {
      const uint64_t idx64 = r * physical_cols + c;
      if (idx64 >= kTileWords)
        return;
      const uint32_t value = tile.words[static_cast<unsigned>(idx64)];

      if (elem_bits == 4u) {
        const uint64_t byte_addr = row_base + (c >> 1u);
        uint8_t packed = dst[byte_addr];
        const uint8_t nibble = static_cast<uint8_t>(value & 0x0fu);
        if ((c & 1u) == 0u)
          packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
        else
          packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4u));
        dst[byte_addr] = packed;
      } else if (elem_bytes == 1u) {
        dst[row_base + c] = static_cast<uint8_t>(value & 0xffu);
      } else if (elem_bytes == 2u) {
        const uint16_t v = static_cast<uint16_t>(value & 0xffffu);
        memcpy(dst + row_base + c * 2u, &v, sizeof(v));
      } else if (elem_bytes == 4u) {
        memcpy(dst + row_base + c * 4u, &value, sizeof(value));
      } else if (elem_bytes == 8u) {
        const uint64_t v = static_cast<uint64_t>(value);
        memcpy(dst + row_base + c * 8u, &v, sizeof(v));
      } else {
        return;
      }
    }
  }
#else
  __builtin_linx_tile_tstore(base, tile, SizeCode, DType, Layout, valid_col,
                             valid_row, physical_col, stride_bytes);
#endif
}

template <unsigned M, unsigned N, unsigned K>
inline RawTile cubeMamulb(RawTile lhs, RawTile rhs) {
  static_assert(M <= 0xffu && N <= 0xffu && K <= 0xffu,
                "PTO Linx canonical v0.57: cube dimensions must fit u8");
#if defined(PTO_HOST_SIM)
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      int64_t acc = 0;
      for (unsigned k = 0; k < K; ++k) {
        const unsigned a_idx = i * K + k;
        const unsigned b_idx = k * N + j;
        if (a_idx >= kTileWords || b_idx >= kTileWords)
          continue;
        const int32_t a = static_cast<int32_t>(lhs.words[a_idx]);
        const int32_t b = static_cast<int32_t>(rhs.words[b_idx]);
        acc += static_cast<int64_t>(a) * static_cast<int64_t>(b);
      }
      const unsigned out_idx = i * N + j;
      if (out_idx < kTileWords)
        out.words[out_idx] = static_cast<uint32_t>(static_cast<int32_t>(acc));
    }
  }
  return out;
#else
  return __builtin_linx_cube_mamulb(lhs, rhs, M, N, K);
#endif
}

template <unsigned M, unsigned N, unsigned K>
inline RawTile cubeMamulbAcc(RawTile acc, RawTile lhs, RawTile rhs) {
  static_assert(M <= 0xffu && N <= 0xffu && K <= 0xffu,
                "PTO Linx canonical v0.57: cube dimensions must fit u8");
#if defined(PTO_HOST_SIM)
  RawTile out = acc;
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      const unsigned out_idx = i * N + j;
      int64_t sum =
          (out_idx < kTileWords) ? static_cast<int32_t>(out.words[out_idx]) : 0;
      for (unsigned k = 0; k < K; ++k) {
        const unsigned a_idx = i * K + k;
        const unsigned b_idx = k * N + j;
        if (a_idx >= kTileWords || b_idx >= kTileWords)
          continue;
        const int32_t a = static_cast<int32_t>(lhs.words[a_idx]);
        const int32_t b = static_cast<int32_t>(rhs.words[b_idx]);
        sum += static_cast<int64_t>(a) * static_cast<int64_t>(b);
      }
      if (out_idx < kTileWords)
        out.words[out_idx] = static_cast<uint32_t>(static_cast<int32_t>(sum));
    }
  }
  return out;
#else
  return __builtin_linx_cube_mamulb_acc(acc, lhs, rhs, M, N, K);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType,
          unsigned SrcDType = DType>
inline RawTile teplUnary(RawTile src, unsigned valid_col, unsigned valid_row,
                         unsigned physical_col) {
  static_assert(TileOpcode <= 0x3ffu,
                "PTO Linx canonical v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned carrier_elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  if (valid_col == 0u || valid_row == 0u || physical_col < valid_col ||
      physical_col == 0u || valid_row > carrier_elems / physical_col)
    return RawTile{};
  RawTile packed{};
  for (unsigned r = 0; r < valid_row; ++r)
    for (unsigned c = 0; c < valid_col; ++c)
      packed.words[r * valid_col + c] = src.words[r * physical_col + c];
  RawTile packed_out = teplUnaryHost<TileOpcode, DType, SrcDType>(
      packed, valid_row * valid_col, valid_row, valid_col);
  RawTile out{};
  if constexpr ((TileOpcode & 0x3ffu) == 0x01du) {
    for (unsigned r = 0; r < valid_col; ++r)
      for (unsigned c = 0; c < valid_row; ++c)
        out.words[r * physical_col + c] = packed_out.words[r * valid_row + c];
  } else {
    for (unsigned r = 0; r < valid_row; ++r)
      for (unsigned c = 0; c < valid_col; ++c)
        out.words[r * physical_col + c] = packed_out.words[r * valid_col + c];
  }
  return out;
#else
  return __builtin_linx_tepl_unary(src, TileOpcode, SizeCode, DType, valid_col,
                                   valid_row, physical_col);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType>
inline RawTile teplBinary(RawTile lhs, RawTile rhs, unsigned valid_col,
                          unsigned valid_row, unsigned physical_col) {
  static_assert(TileOpcode <= 0x3ffu,
                "PTO Linx canonical v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned carrier_elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  if (valid_col == 0u || valid_row == 0u || physical_col < valid_col ||
      physical_col == 0u || valid_row > carrier_elems / physical_col)
    return RawTile{};
  RawTile packed_lhs{};
  RawTile packed_rhs{};
  for (unsigned r = 0; r < valid_row; ++r)
    for (unsigned c = 0; c < valid_col; ++c) {
      packed_lhs.words[r * valid_col + c] = lhs.words[r * physical_col + c];
      packed_rhs.words[r * valid_col + c] = rhs.words[r * physical_col + c];
    }
  RawTile packed_out = teplBinaryHost<TileOpcode, DType>(packed_lhs, packed_rhs,
                                                         valid_row * valid_col);
  RawTile out{};
  for (unsigned r = 0; r < valid_row; ++r)
    for (unsigned c = 0; c < valid_col; ++c)
      out.words[r * physical_col + c] = packed_out.words[r * valid_col + c];
  return out;
#else
  return __builtin_linx_tepl_binary(lhs, rhs, TileOpcode, SizeCode, DType,
                                    valid_col, valid_row, physical_col);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType, unsigned Mode,
          typename Scalar>
inline RawTile teplBinaryScalar(RawTile lhs, Scalar scalar, unsigned valid_col,
                                unsigned valid_row, unsigned physical_col) {
  static_assert(TileOpcode <= 0x3ffu,
                "PTO Linx canonical v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
  static_assert(Mode == 1u, "PTO Linx canonical v0.57: tepl.binary.scalar "
                            "requires operand mode=VS(1)");
#if defined(PTO_HOST_SIM)
  RawTile rhs{};
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  const long long bits = encodeScalar(scalar);
  const uint32_t scalar_word = scalarToWordDType(bits, DType);
  for (unsigned i = 0; i < elems && i < kTileWords; ++i)
    rhs.words[i] = scalar_word;
  return teplBinary<TileOpcode, SizeCode, DType>(lhs, rhs, valid_col, valid_row,
                                                 physical_col);
#else
  return __builtin_linx_tepl_binary_scalar(lhs, encodeScalar(scalar),
                                           TileOpcode, SizeCode, DType, Mode,
                                           valid_col, valid_row, physical_col);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType, unsigned Mode,
          typename Scalar>
inline RawTile teplSplat(Scalar scalar, unsigned valid_col, unsigned valid_row,
                         unsigned physical_col) {
  static_assert(TileOpcode <= 0x3ffu,
                "PTO Linx canonical v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
  static_assert(
      Mode == 2u,
      "PTO Linx canonical v0.57: tepl.splat requires operand mode=SV(2)");
#if defined(PTO_HOST_SIM)
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);

  if ((TileOpcode & 0x3ffu) != 0x019u)
    return out;

  const long long bits = encodeScalar(scalar);
  const uint32_t scalar_word = scalarToWordDType(bits, DType);
  if (valid_col == 0u || valid_row == 0u || physical_col < valid_col ||
      physical_col == 0u || valid_row > elems / physical_col)
    return RawTile{};
  for (unsigned r = 0; r < valid_row; ++r)
    for (unsigned c = 0; c < valid_col; ++c)
      out.words[r * physical_col + c] = scalar_word;
  return out;
#else
  return __builtin_linx_tepl_splat(encodeScalar(scalar), TileOpcode, SizeCode,
                                   DType, Mode, valid_col, valid_row,
                                   physical_col);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout,
          unsigned HasLayout, unsigned Mode>
inline RawTile tileTMov(RawTile src) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "PTO Linx canonical v0.57: size_code must be in [5,8]");
  static_assert(HasLayout <= 1u,
                "PTO Linx canonical v0.57: has_layout must be bool");
  static_assert(Mode <= 1u,
                "PTO Linx canonical v0.57: tmov mode must be 0(V2V) or 1(A2V)");
#if defined(PTO_HOST_SIM)
  (void)DType;
  (void)Layout;
  (void)HasLayout;
  (void)Mode;
  return src;
#else
  return __builtin_linx_tile_tmov(src, Mode, SizeCode, DType, Layout,
                                  HasLayout);
#endif
}

} // namespace detail
} // namespace linx
} // namespace pto

#endif // PTO_LINX_IMPL_BACKEND_HPP
