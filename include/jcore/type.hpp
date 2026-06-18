#ifndef _INCLUDE_JCORE_TYPE_H_
#define _INCLUDE_JCORE_TYPE_H_

#include <type_traits>
#include <cstddef>
enum __type_code {
  __type_fp64 = 0,
  __type_fp32 = 1,

  __type_fp16 = 4,
  __type_bf16 = 5,

  __type_fp8_e4m3 = 7,
  __type_fp8_e5m2 = 8,

  __type_int64 = 16,
  __type_int32 = 17,
  __type_int16 = 18,
  __type_int8 = 19,

  __type_uint64 = 24,
  __type_uint32 = 25,
  __type_uint16 = 26,
  __type_uint8 = 27,
};

template <int C> struct type_traits_base {
  static constexpr int TypeCode = C;
};

template <typename E> struct type_traits {};

// clang-format off
template<> struct type_traits<double>         : public type_traits_base<__type_fp64> {};
template<> struct type_traits<__fp32>         : public type_traits_base<__type_fp32> {};
template<> struct type_traits<__half>         : public type_traits_base<__type_fp16> {};
template<> struct type_traits<__bf16>         : public type_traits_base<__type_bf16> {};
template<> struct type_traits<__fp8_e4m3>     : public type_traits_base<__type_fp8_e4m3> {};
template<> struct type_traits<__fp8_e5m2>     : public type_traits_base<__type_fp8_e5m2> {};
template<> struct type_traits<int64_t>        : public type_traits_base<__type_int64> {};
template<> struct type_traits<int32_t>        : public type_traits_base<__type_int32> {};
template<> struct type_traits<int16_t>        : public type_traits_base<__type_int16> {};
template<> struct type_traits<int8_t>         : public type_traits_base<__type_int8> {};
template<> struct type_traits<unsigned long>  : public type_traits_base<__type_uint64> {};
template<> struct type_traits<unsigned int>   : public type_traits_base<__type_uint32> {};
template<> struct type_traits<unsigned short> : public type_traits_base<__type_uint16> {};
template<> struct type_traits<unsigned char>  : public type_traits_base<__type_uint8> {};

// Add uint32_t specialization - needed because uint32_t is a typedef that may not match unsigned int
#include <cstdint>
template<> struct type_traits<uint32_t>       : public type_traits_base<__type_uint32> {};
template<> struct type_traits<uint64_t>       : public type_traits_base<__type_uint64> {};
template<> struct type_traits<uint16_t>       : public type_traits_base<__type_uint16> {};
template<> struct type_traits<uint8_t>        : public type_traits_base<__type_uint8> {};
// clang-format on


enum __tilesize_code{
  __tilesize_16B   = 0,
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
      b == 16     ? __tilesize_16B  :
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
