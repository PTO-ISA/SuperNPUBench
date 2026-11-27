#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "SuperNPUBench kernel argument ABI requires little-endian host and device"
#endif

namespace supernpu {

template <std::size_t Bytes>
inline void CopyKernelArgumentBits(void* destination, const void* source) {
#if defined(__linx_v5) && defined(__clang__)
    __builtin_memcpy_inline(destination, source, Bytes);
#else
    std::memcpy(destination, source, Bytes);
#endif
}

constexpr std::uint32_t kKernelArgumentMagic = 0x534e5055u;  // "SNPU"
constexpr std::uint16_t kKernelArgumentVersion = 1;

enum class KernelArgumentType : std::uint32_t {
    Buffer = 1,
    U32 = 2,
    I32 = 3,
    U64 = 4,
    I64 = 5,
    F32 = 6,
    F64 = 7,
};

enum class KernelArgumentStatus : std::uint64_t {
    Success = 0,
    NullDescriptor = 0x100,
    MisalignedDescriptor = 0x101,
    InvalidMagic = 0x102,
    UnsupportedVersion = 0x103,
    InvalidHeaderSize = 0x104,
    InvalidPayloadSize = 0x105,
    SchemaMismatch = 0x106,
    TypeMismatch = 0x107,
};

struct KernelArgumentHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t headerBytes;
    std::uint32_t payloadBytes;
    std::uint32_t schema;
};

struct KernelArgumentRecord {
    std::uint32_t type;
    std::uint32_t reserved;
    std::uint64_t value;
};

static_assert(sizeof(KernelArgumentHeader) == 16,
              "kernel argument header ABI must remain 16 bytes");
static_assert(sizeof(KernelArgumentRecord) == 16,
              "kernel argument record ABI must remain 16 bytes");
static_assert(offsetof(KernelArgumentHeader, magic) == 0 &&
                  offsetof(KernelArgumentHeader, version) == 4 &&
                  offsetof(KernelArgumentHeader, headerBytes) == 6 &&
                  offsetof(KernelArgumentHeader, payloadBytes) == 8 &&
                  offsetof(KernelArgumentHeader, schema) == 12,
              "kernel argument header field offsets changed");
static_assert(offsetof(KernelArgumentRecord, type) == 0 &&
                  offsetof(KernelArgumentRecord, reserved) == 4 &&
                  offsetof(KernelArgumentRecord, value) == 8,
              "kernel argument record field offsets changed");
static_assert(alignof(KernelArgumentRecord) == alignof(std::uint64_t),
              "kernel argument record ABI requires 64-bit alignment");
static_assert(std::is_standard_layout<KernelArgumentHeader>::value &&
                  std::is_trivially_copyable<KernelArgumentHeader>::value,
              "kernel argument header must have a stable object representation");
static_assert(std::is_standard_layout<KernelArgumentRecord>::value &&
                  std::is_trivially_copyable<KernelArgumentRecord>::value,
              "kernel argument record must have a stable object representation");

inline KernelArgumentRecord MakeKernelArgument(KernelArgumentType type,
                                                std::uint64_t value) {
    return {static_cast<std::uint32_t>(type), 0, value};
}

inline std::uint64_t KernelArgumentBits(float value) {
    std::uint32_t bits;
    CopyKernelArgumentBits<sizeof(bits)>(&bits, &value);
    return bits;
}

inline std::uint64_t KernelArgumentBits(double value) {
    std::uint64_t bits;
    CopyKernelArgumentBits<sizeof(bits)>(&bits, &value);
    return bits;
}

inline float KernelArgumentF32(std::uint64_t bits) {
    const std::uint32_t narrow = static_cast<std::uint32_t>(bits);
    float value;
    CopyKernelArgumentBits<sizeof(value)>(&value, &narrow);
    return value;
}

inline std::int32_t KernelArgumentI32(std::uint64_t bits) {
    const std::uint32_t narrow = static_cast<std::uint32_t>(bits);
    std::int32_t value;
    CopyKernelArgumentBits<sizeof(value)>(&value, &narrow);
    return value;
}

inline std::int64_t KernelArgumentI64(std::uint64_t bits) {
    std::int64_t value;
    CopyKernelArgumentBits<sizeof(value)>(&value, &bits);
    return value;
}

inline double KernelArgumentF64(std::uint64_t bits) {
    double value;
    CopyKernelArgumentBits<sizeof(value)>(&value, &bits);
    return value;
}

}  // namespace supernpu
