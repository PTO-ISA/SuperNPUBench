#pragma once
#include <cstdint>

namespace hif4 {
inline uint16_t bf16_bits(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffff) > 0x7f800000) return (bits >> 16) | 0x40;
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
inline float bf16_value(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16;
    float value;
    __builtin_memcpy(&value, &word, sizeof(value));
    return value;
}
// Non-negative BF16 -> E6M2 (bias 48, no subnormal/zero, 255 = NaN).
inline uint8_t e6m2_from_bf16(uint16_t bits) {
    if (bits >= 0x7f80) return 255;
    unsigned rounded = bits + 15 + ((bits >> 5) & 1);
    if (rounded < (79u << 7)) return 0;
    unsigned code = (rounded >> 5) - (79u << 2);
    return code > 254 ? 254 : code;
}

// Encode a BF16 bit pattern as a signed quarter-step S1P2 lane. RNE ties
// choose an even magnitude; overflow saturates. NaN has no S1P2 encoding:
// use zero payload (the accompanying E6M2 scale carries NaN).
inline std::uint8_t s1p2_from_bf16(std::uint16_t bits) {
    const unsigned sign = (bits >> 12) & 8u;
    const unsigned exponent = (bits >> 7) & 255u;
    const unsigned fraction = bits & 127u;
    if (exponent == 255 && fraction) return 0;
    if (exponent > 127) return static_cast<std::uint8_t>(sign | 7u);
    if (exponent < 124) return static_cast<std::uint8_t>(sign);
    const unsigned shift = 132u - exponent;
    const unsigned significand = 128u + fraction;
    unsigned magnitude = significand >> shift;
    const unsigned remainder = significand & ((1u << shift) - 1u);
    const unsigned halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (magnitude & 1u)))
        ++magnitude;
    if (magnitude > 7u) magnitude = 7u;
    return static_cast<std::uint8_t>(sign | magnitude);
}
} // namespace hif4
