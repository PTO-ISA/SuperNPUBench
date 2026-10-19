#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                       \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "elementwise_masked_gather_scatter requires the typed element API"
#endif

#include <cstddef>
#include <cstdint>

#ifndef ELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE
#define ELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE 0
#endif

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElements = 128;
constexpr std::size_t kPartElements = 32;

// 保留原始 byte mask 分配；公共 element API 使用 U32 谓词承载，main 仅将
// byte 值拓宽到这个 transport 数组。
alignas(32) static uint32_t mask_transport[kElements];

__attribute__((noinline)) void elementwise_masked_gather_scatter(
    float *__restrict output, const float *__restrict source,
    const int32_t *__restrict indices, const uint32_t *__restrict mask) {
  ElementTile<int32_t, kElements> index_tile;
  ElementTile<uint32_t, kElements> mask_tile;
  TLOAD(index_tile, indices, kElements);
  TLOAD(mask_tile, mask, kElements);

  auto index_parts = TPARTVIEW<kPartElements>(index_tile, kElements);
  auto mask_parts = TPARTVIEW<kPartElements>(mask_tile, kElements);

  for (std::size_t part = 0; part < index_parts.size(); ++part) {
    ElementTile<int32_t, kPartElements> part_indices;
    ElementTile<uint32_t, kPartElements> part_mask;
    ElementTile<float, kPartElements> gathered;
    ElementTile<float, kPartElements> transported;

    auto index_part = index_parts.part(part);
    auto mask_part = mask_parts.part(part);
    TMOV(part_indices, index_part);
    TMOV(part_mask, mask_part);
    auto &index_elements = TPARTELEMENT(part_indices);
    auto &mask_elements = TPARTELEMENT(part_mask);
    auto &gathered_elements = TPARTELEMENT(gathered);

#pragma pto element for
    for (unsigned element = 0; element < kPartElements; ++element) {
      if (mask_elements[element] != 0u)
        gathered_elements[element] = source[index_elements[element]];
      else
        gathered_elements[element] = 0.0f;
    }

    // 用原始位 Tile transport 隔开两个保持源码顺序的 element 区域。
    TMOV(transported, gathered);
    auto &transported_elements = TPARTELEMENT(transported);

#pragma pto element for
    for (unsigned element = 0; element < kPartElements; ++element) {
      if (mask_elements[element] != 0u)
        output[index_elements[element]] = transported_elements[element];
    }
  }
}

extern "C" {
alignas(32) float source_storage[kElements] =
    {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
alignas(32) int32_t index_storage[kElements] = {0, 1, 2, 3, 4, 5, 6, 7};
alignas(32) unsigned char mask_storage[kElements] =
    {1, 0, 1, 0, 1, 0, 1, 0};
alignas(32) float gathered_storage[kElements];
#if ELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE
alignas(32) unsigned char output_guard_before[64];
#endif
alignas(32) float output_storage[kElements];
#if ELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE
alignas(32) unsigned char output_guard_after[64];
#endif
}

__attribute__((optnone)) int main() {
#if ELEMENTWISE_MASKED_GATHER_SCATTER_STRONG_CASE
  const uint32_t source_pattern[16] = {
      0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u,
      0x7fc12345u, 0xffc54321u, 0x3f800000u, 0xbf800000u,
      0x7f800123u, 0xff800321u, 0x00000001u, 0x80000001u,
      0x7f7fffffu, 0xff7fffffu, 0x3eaaaaabu, 0xbeaaaaabu,
  };
  const int32_t active_indices[16] = {
      -64, -63, -62, -61, -60, -59, -58, -57,
      8,   9,   10,  11,  12,  13,  14,  15,
  };
  const unsigned char active_masks[4] = {2u, 3u, 0x80u, 0xffu};
  const int32_t poison_indices[4] = {
      INT32_MIN, INT32_MAX, INT32_C(-1073741824), INT32_C(1073741824)};
  const uint32_t sentinel = 0x5a5a6b6bu;

  for (std::size_t element = 0; element < kElements; ++element) {
    const uint32_t bits = source_pattern[element % 16];
    __builtin_memcpy(source_storage + element, &bits, sizeof(bits));
    __builtin_memcpy(output_storage + element, &sentinel, sizeof(sentinel));
    gathered_storage[element] = 0.0f;
    if ((element & 3u) < 2u) {
      index_storage[element] = active_indices[(element / 4u) % 16u];
      mask_storage[element] = active_masks[(element / 4u) % 4u];
    } else {
      index_storage[element] = poison_indices[(element / 4u) % 4u];
      mask_storage[element] = 0u;
    }
  }
  for (unsigned byte = 0; byte < 64; ++byte) {
    output_guard_before[byte] = 0x5au;
    output_guard_after[byte] = 0xa5u;
  }
  float *source_base = source_storage + 64;
  float *output_base = output_storage + 64;
#else
  const float *source_base = source_storage;
  float *output_base = output_storage;
#endif

  for (std::size_t element = 0; element < kElements; ++element)
    mask_transport[element] = static_cast<uint32_t>(mask_storage[element]);

  elementwise_masked_gather_scatter(output_base, source_base, index_storage,
                                    mask_transport);
  return 0;
}
