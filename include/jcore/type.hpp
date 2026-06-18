#ifndef _INCLUDE_JCORE_TYPE_H_
#define _INCLUDE_JCORE_TYPE_H_

#include <type_traits>
#include <cstddef>

enum __type_code {
  __type_fp64 = 0,
  __type_fp32 = 1,
  __type_tf32 = 2,
  __type_hf32 = 3,

  __type_fp16 = 4,
  __type_bf16 = 5,
  __type_hif8 = 6,

  __type_fp8_e4m3 = 7,
  __type_fp8_e5m2 = 8,
  __type_fp6_e3m2 = 9,
  __type_fp5_e2m3 = 10,

  __type_fp4_e2m1x2 = 11,
  __type_fp4_e1m2x2 = 12,
  __type_fp8_e8m0 = 13,

  __type_fp4_hif4x2 = 14,

  __type_int64 = 16,
  __type_int32 = 17,
  __type_int16 = 18,
  __type_int8 = 19,
  __type_int4x2 = 20,

  __type_uint64 = 24,
  __type_uint32 = 25,
  __type_uint16 = 26,
  __type_uint8 = 27,
  __type_uint4x2 = 28,
};

template <int C, int b> struct type_traits_base {
  static constexpr int TypeCode = C;
  static constexpr int bits = b;
};

// clang-format off
template<> struct type_traits<double>         : public type_traits_base<__type_fp64, 64> {};
template<> struct type_traits<__fp32>         : public type_traits_base<__type_fp32, 32> {};
template<> struct type_traits<__tf32>         : public type_traits_base<__type_tf32, 32> {};
template<> struct type_traits<__hf32>         : public type_traits_base<__type_hf32, 32> {};

template<> struct type_traits<__half>         : public type_traits_base<__type_fp16, 16> {};
template<> struct type_traits<__bf16>         : public type_traits_base<__type_bf16, 16> {};
template<> struct type_traits<__hif8>         : public type_traits_base<__type_hif8, 8> {};

template<> struct type_traits<__fp8_e4m3>     : public type_traits_base<__type_fp8_e4m3, 8> {};
template<> struct type_traits<__fp8_e5m2>     : public type_traits_base<__type_fp8_e5m2, 8> {};
template<> struct type_traits<__fp6_e3m2>     : public type_traits_base<__type_fp6_e3m2, 6> {};
template<> struct type_traits<__fp6_e2m3>     : public type_traits_base<__type_fp5_e2m3, 6> {};
template<> struct type_traits<__fp4_e2m1x2>   : public type_traits_base<__type_fp4_e2m1x2, 8> {};
template<> struct type_traits<__fp4_e1m2x2>   : public type_traits_base<__type_fp4_e1m2x2, 8> {};
template<> struct type_traits<__fp8_e8m0>     : public type_traits_base<__type_fp8_e8m0, 8> {};
template<> struct type_traits<__fp4_hif4x2>   : public type_traits_base<__type_fp4_hif4x2, 8> {};

template<> struct type_traits<int64_t>        : public type_traits_base<__type_int64, 64> {};
template<> struct type_traits<int32_t>        : public type_traits_base<__type_int32, 32> {};
template<> struct type_traits<int16_t>        : public type_traits_base<__type_int16, 16> {};
template<> struct type_traits<int8_t>         : public type_traits_base<__type_int8, 8> {};
template<> struct type_traits<__int4x2>       : public type_traits_base<__type_int4x2, 8> {};

template<> struct type_traits<unsigned long>  : public type_traits_base<__type_uint64, 64> {};
template<> struct type_traits<unsigned int>   : public type_traits_base<__type_uint32, 32> {};
template<> struct type_traits<unsigned short> : public type_traits_base<__type_uint16, 16> {};
template<> struct type_traits<unsigned char>  : public type_traits_base<__type_uint8, 8> {};
template<> struct type_traits<__uint4x2>      : public type_traits_base<__type_uint4x2, 8> {};
// clang-format on


enum __tilesize_code{
  __tilesize_0B    = 0,
  __tilesize_32B   = 1,
  __tilesize_64B   = 2,
  __tilesize_128B  = 3,
  __tilesize_256B  = 4,
  __tilesize_512B  = 5,
  __tilesize_1KB   = 6,
  __tilesize_2KB   = 7,
  __tilesize_4KB   = 8,
  __tilesize_8KB   = 9,
  __tilesize_16KB  = 10,
  __tilesize_32KB  = 11,
  __tilesize_64KB  = 12,
  __tilesize_128KB = 13,
  __tilesize_256KB = 14,
  __tilesize_512KB = 15,
  __tilesize_unknown = -1
};

template <typename T>
struct tile_type_traits {
private:
  using PlainT = std::remove_cv_t<std::remove_reference_t<T>>;
  static constexpr std::size_t bytes = sizeof(PlainT);

  static constexpr int mapBytesToEnum(std::size_t b) {
    return
      b == 0      ? __tilesize_0B   :
      b == 32     ? __tilesize_32B  :
      b == 64     ? __tilesize_64B  :
      b == 128    ? __tilesize_128B :
      b == 256    ? __tilesize_256B :
      b == 512    ? __tilesize_512B :
      b == 1024   ? __tilesize_1KB  :
      b == 2048   ? __tilesize_2KB  :
      b == 4096   ? __tilesize_4KB  :
      b == 8192   ? __tilesize_8KB  :
      b == 16384  ? __tilesize_16KB :
      b == 32768  ? __tilesize_32KB :
      b == 65536  ? __tilesize_64KB :
      b == 131072 ? __tilesize_128KB:
      b == 262144 ? __tilesize_256KB:
      b == 524288 ? __tilesize_512KB:
      __tilesize_unknown;
  }

public:
  static constexpr int TilesizeCode = mapBytesToEnum(bytes);
  static constexpr int Regsize = bytes;
};

#endif
