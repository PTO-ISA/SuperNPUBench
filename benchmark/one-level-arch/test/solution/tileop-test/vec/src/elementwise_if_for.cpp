#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) || \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "elementwise_if_for requires the official typed element API"
#endif

#include <cstddef>
#include <cstdint>

#ifndef ELEMENTWISE_IF_FOR_STRONG_CASE
#define ELEMENTWISE_IF_FOR_STRONG_CASE 0
#endif

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

// One complete kernel: Tile loads, ordinary per-element control flow, then
// Tile stores. Logical partitions keep physical layout inside the public API.
__attribute__((noinline)) void elementwise_if_for(float *output,
                                                 const float *lhs,
                                                 const float *rhs) {
  ElementTile<float, 128> a;
  ElementTile<float, 128> b;
  TLOAD(a, lhs, 128);
  TLOAD(b, rhs, 128);
  auto a_parts = TPARTVIEW<32>(a, 128);
  auto b_parts = TPARTVIEW<32>(b, 128);

  for (std::size_t part = 0; part < a_parts.size(); ++part) {
    ElementTile<float, 32> a_part;
    ElementTile<float, 32> b_part;
    ElementTile<float, 32> result;
    auto a_view = a_parts.part(part);
    auto b_view = b_parts.part(part);
    TMOV(a_part, a_view);
    TMOV(b_part, b_view);
    auto &a_elements = TPARTELEMENT(a_part);
    auto &b_elements = TPARTELEMENT(b_part);
    auto &result_elements = TPARTELEMENT(result);

#pragma pto element for
    for (unsigned element = 0; element < 32; ++element) {
      if (a_elements[element] > 0.0f)
        result_elements[element] = a_elements[element] + b_elements[element];
      else
        result_elements[element] = a_elements[element] - b_elements[element];
    }

    TSTORE(output, result, a_parts, part);
  }
}

// The original case retains exactly three aligned 512-byte, zero-initialized
// buffers. The separate strong case adds output guards and finite bit patterns.
extern "C" {
alignas(32) unsigned char lhs_storage[512];
alignas(32) unsigned char rhs_storage[512];
#if ELEMENTWISE_IF_FOR_STRONG_CASE
alignas(32) unsigned char out_storage[640];
#else
alignas(32) unsigned char out_storage[512];
#endif
}

__attribute__((optnone)) int main() {
#if ELEMENTWISE_IF_FOR_STRONG_CASE
  // Exact dyadic values, both branch arms, cancellation, and all signed-zero
  // pairs. Initialization uses object bytes, without arithmetic conversions.
  const uint32_t lhs_bits[16] = {
      0x00000000, 0x80000000, 0x00000000, 0x80000000,
      0x3f800000, 0xbf800000, 0x3f800000, 0xbf800000,
      0x3f000000, 0xbf000000, 0x40000000, 0xc0000000,
      0x3e800000, 0xbe800000, 0x41800000, 0xc1800000};
  const uint32_t rhs_bits[16] = {
      0x00000000, 0x00000000, 0x80000000, 0x80000000,
      0xbf800000, 0xbf800000, 0xc0000000, 0x40000000,
      0x3e800000, 0x3e800000, 0x3f800000, 0xbf800000,
      0xbe000000, 0xbe000000, 0xc1000000, 0x41000000};
  for (unsigned element = 0; element < 128; ++element) {
    __builtin_memcpy(lhs_storage + element * 4, &lhs_bits[element % 16], 4);
    __builtin_memcpy(rhs_storage + element * 4, &rhs_bits[element % 16], 4);
  }
  for (unsigned byte = 0; byte < sizeof(out_storage); ++byte)
    out_storage[byte] = 0x5a;
  auto *out = reinterpret_cast<float *>(out_storage + 64);
#else
  auto *out = reinterpret_cast<float *>(out_storage);
#endif
  elementwise_if_for(out, reinterpret_cast<const float *>(lhs_storage),
                     reinterpret_cast<const float *>(rhs_storage));
  return 0;
}
