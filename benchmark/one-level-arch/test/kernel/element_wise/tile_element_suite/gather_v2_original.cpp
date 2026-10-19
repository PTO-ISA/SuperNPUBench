#include <common/pto_tileop.hpp>

#include "benchmark.h"
#include "solution/gather_v2/gather_v2.hpp"

#include <cstdint>

#ifndef GATHER_V2_PROFILE
#define GATHER_V2_PROFILE 0
#endif

#ifndef GATHER_V2_ALIAS_PROBE
#define GATHER_V2_ALIAS_PROBE 0
#endif

namespace {

#if GATHER_V2_PROFILE == 0
using Value = float;
using Bits = uint32_t;
using Index = uint32_t;
using IndexBits = uint32_t;
constexpr uint32_t kRank = 1, kGatherDim = 0;
constexpr uint32_t kInputElements = 19, kOutputElements = 37;
constexpr uint32_t kIndexElements = 37, kTileElements = 32;
constexpr uint32_t kInputShape[2] = {19, 1};
constexpr uint32_t kOutputShape[2] = {37, 1};
#elif GATHER_V2_PROFILE == 1
using Value = int32_t;
using Bits = uint32_t;
using Index = int32_t;
using IndexBits = uint32_t;
constexpr uint32_t kRank = 2, kGatherDim = 1;
constexpr uint32_t kInputElements = 55, kOutputElements = 35;
constexpr uint32_t kIndexElements = 7, kTileElements = 32;
constexpr uint32_t kInputShape[2] = {5, 11};
constexpr uint32_t kOutputShape[2] = {5, 7};
#elif GATHER_V2_PROFILE == 2
using Value = __half;
using Bits = uint16_t;
using Index = uint32_t;
using IndexBits = uint32_t;
constexpr uint32_t kRank = 2, kGatherDim = 0;
constexpr uint32_t kInputElements = 187, kOutputElements = 136;
constexpr uint32_t kIndexElements = 8, kTileElements = 64;
constexpr uint32_t kInputShape[2] = {11, 17};
constexpr uint32_t kOutputShape[2] = {8, 17};
#else
#error unsupported original gather_v2 profile
#endif

constexpr bool kAlias = GATHER_V2_ALIAS_PROBE != 0;
static_assert(!kAlias || GATHER_V2_PROFILE == 1,
              "the race-free alias probe uses the S32 axis-1 profile");
constexpr uint32_t kIndexPadding = 128 / sizeof(Index);
constexpr uint32_t kInputCount = kInputElements + kTileElements;
constexpr uint32_t kIndexCount = kIndexElements + kIndexPadding;
constexpr uint32_t kOutputCount = kOutputElements + kTileElements;
constexpr uint32_t kAliasCount = kInputCount;
constexpr uint32_t kBeforeGuardCount = 4096 / sizeof(Value);
constexpr uint32_t kAfterGuardCount = 64 / sizeof(Value);
constexpr uint32_t kIndexBeforeGuardCount = 4096 / sizeof(Index);
constexpr uint32_t kIndexAfterGuardCount = 64 / sizeof(Index);
constexpr Bits kSentinel = static_cast<Bits>(0x5a5a5a5au);
constexpr IndexBits kIndexSentinel = 0x5a5a5a5au;

template <typename T, uint32_t Count> struct Buffer { T data[Count]; };

constexpr Bits input_bits(uint32_t index) {
  if constexpr (sizeof(Bits) == 2)
    return static_cast<Bits>(index * 40503u + 0x35a7u);
  if constexpr (GATHER_V2_PROFILE == 0) {
    constexpr uint32_t edge[] = {
        0u, 0x80000000u, 0x7f800000u, 0xff800000u,
        0x7fc12345u, 0xffc54321u, 1u, 0x80000001u};
    if (index < 8)
      return edge[index];
  }
  return index * 104729u + 0x80003039u;
}

constexpr IndexBits index_bits(uint32_t index) {
  if constexpr (GATHER_V2_PROFILE == 1) {
    constexpr uint32_t values[] = {4, 5, 6, 7, 8, 2, 2};
    return values[index];
  }
  if constexpr (GATHER_V2_PROFILE == 2)
    if (index + 1 == kIndexElements)
      return (2u * 7u + 3u) % kInputShape[kGatherDim];
  return (index * 7u + 3u) % kInputShape[kGatherDim];
}

constexpr Value value_from_bits(Bits bits) {
  return __builtin_bit_cast(Value, bits);
}

constexpr Index index_from_bits(IndexBits bits) {
  return __builtin_bit_cast(Index, bits);
}

template <uint32_t Count>
consteval Buffer<Value, Count> value_sentinels() {
  Buffer<Value, Count> result{};
  for (uint32_t i = 0; i < Count; ++i)
    result.data[i] = value_from_bits(kSentinel);
  return result;
}

template <uint32_t Count>
consteval Buffer<Index, Count> index_sentinels() {
  Buffer<Index, Count> result{};
  for (uint32_t i = 0; i < Count; ++i)
    result.data[i] = index_from_bits(kIndexSentinel);
  return result;
}

consteval Buffer<Value, kInputCount> input_fixture() {
  auto result = value_sentinels<kInputCount>();
  for (uint32_t i = 0; i < kInputElements; ++i)
    result.data[i] = value_from_bits(input_bits(i));
  return result;
}

consteval Buffer<Index, kIndexCount> index_fixture() {
  auto result = index_sentinels<kIndexCount>();
  for (uint32_t i = 0; i < kIndexElements; ++i)
    result.data[i] = index_from_bits(index_bits(i));
  return result;
}

consteval Buffer<Value, kAliasCount> alias_fixture() {
  auto result = value_sentinels<kAliasCount>();
  for (uint32_t i = 0; i < kInputElements; ++i)
    result.data[i] = value_from_bits(input_bits(i));
  return result;
}

} // namespace

extern "C" {
alignas(4096) Buffer<Value, kBeforeGuardCount> gather_v2_input_guard_before =
    value_sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<Value, kInputCount> gather_v2_input = input_fixture();
alignas(sizeof(Value)) Buffer<Value, kAfterGuardCount>
    gather_v2_input_guard_after = value_sentinels<kAfterGuardCount>();

alignas(4096) Buffer<Index, kIndexBeforeGuardCount>
    gather_v2_index_guard_before = index_sentinels<kIndexBeforeGuardCount>();
alignas(4096) Buffer<Index, kIndexCount> gather_v2_index = index_fixture();
alignas(sizeof(Index)) Buffer<Index, kIndexAfterGuardCount>
    gather_v2_index_guard_after = index_sentinels<kIndexAfterGuardCount>();

alignas(4096) Buffer<Value, kBeforeGuardCount> gather_v2_output_guard_before =
    value_sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<Value, kOutputCount> gather_v2_output =
    value_sentinels<kOutputCount>();
alignas(sizeof(Value)) Buffer<Value, kAfterGuardCount>
    gather_v2_output_guard_after = value_sentinels<kAfterGuardCount>();

alignas(4096) Buffer<Value, kBeforeGuardCount> gather_v2_alias_guard_before =
    value_sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<Value, kAliasCount> gather_v2_alias = alias_fixture();
alignas(sizeof(Value)) Buffer<Value, kAfterGuardCount>
    gather_v2_alias_guard_after = value_sentinels<kAfterGuardCount>();

alignas(32) uint32_t gather_v2_seen[4];
alignas(32) uint32_t gather_v2_status[20] = {
    GATHER_V2_PROFILE, kRank, kGatherDim, kInputElements, kOutputElements,
    kIndexElements, kTileElements, kInputShape[0], kInputShape[1],
    kOutputShape[0], kOutputShape[1], kInputCount, kIndexCount, kOutputCount,
    kAlias, sizeof(Value), sizeof(Index), 4, 0x47563231u, 0};
}

int main() {
  const uint32_t pe = get_thread_idx();
  if (pe >= 4)
    return 1;

  uint32_t input_shape[kRank];
  uint32_t output_shape[kRank];
  for (uint32_t dim = 0; dim < kRank; ++dim) {
    input_shape[dim] = kInputShape[dim];
    output_shape[dim] = kOutputShape[dim];
  }

  Value *input = kAlias ? gather_v2_alias.data : gather_v2_input.data;
  Value *output = kAlias ? gather_v2_alias.data : gather_v2_output.data;

  BENCHSTART;
  [[clang::noinline]]
  supernpu::tile_isa::gather_v2<
      Value, Index, kRank, kGatherDim, kInputElements, kOutputElements,
      kTileElements>(input, gather_v2_index.data, output, input_shape,
                     output_shape);
  BENCHEND;

  gather_v2_seen[pe] = 0x47560000u | pe;
  return 0;
}
