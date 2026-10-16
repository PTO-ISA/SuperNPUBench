#include <common/pto_tileop.hpp>

#include "benchmark.h"
#include "solution/view_copy/view_copy.hpp"

#include <cstdint>

#ifndef VIEW_COPY_PROFILE
#define VIEW_COPY_PROFILE 0
#endif

#ifndef VIEW_COPY_ALIAS_PROBE
#define VIEW_COPY_ALIAS_PROBE 0
#endif

namespace {

#if VIEW_COPY_PROFILE == 0
using Value = int32_t;
using Bits = uint32_t;
constexpr uint32_t kElements = 64;
constexpr uint32_t kTileElements = 32;
constexpr uint32_t kShape[3] = {2, 4, 8};
constexpr uint32_t kInputStride[3] = {32, 1, 4};
constexpr uint32_t kOutputStride[3] = {32, 8, 1};
constexpr uint32_t kInputOffsetBytes = 32;
constexpr uint32_t kOutputOffsetBytes = 64;
#elif VIEW_COPY_PROFILE == 1
using Value = __half;
using Bits = uint16_t;
constexpr uint32_t kElements = 128;
constexpr uint32_t kTileElements = 64;
constexpr uint32_t kShape[3] = {4, 4, 8};
constexpr uint32_t kInputStride[3] = {32, 8, 1};
constexpr uint32_t kOutputStride[3] = {32, 1, 4};
constexpr uint32_t kInputOffsetBytes = 64;
constexpr uint32_t kOutputOffsetBytes = 128;
#elif VIEW_COPY_PROFILE == 2
using Value = __half;
using Bits = uint16_t;
constexpr uint32_t kElements = 96;
constexpr uint32_t kTileElements = 64;
constexpr uint32_t kShape[3] = {4, 4, 6};
constexpr uint32_t kInputStride[3] = {24, 1, 4};
constexpr uint32_t kOutputStride[3] = {24, 6, 1};
constexpr uint32_t kInputOffsetBytes = 64;
constexpr uint32_t kOutputOffsetBytes = 96;
#else
#error unsupported original view_copy profile
#endif

constexpr bool kAlias = VIEW_COPY_ALIAS_PROBE != 0;
static_assert(!kAlias || VIEW_COPY_PROFILE == 0,
              "the deterministic alias probe uses the S32 profile");
constexpr uint32_t kInputOffsetElements = kInputOffsetBytes / sizeof(Value);
constexpr uint32_t kOutputOffsetElements = kOutputOffsetBytes / sizeof(Value);
constexpr uint32_t kInputCount = kInputOffsetElements + kElements;
constexpr uint32_t kOutputCount = kOutputOffsetElements + kElements;
constexpr uint32_t kAliasView = kOutputOffsetElements;
constexpr uint32_t kAliasInputAdjustment =
    kOutputOffsetElements - kInputOffsetElements;
constexpr uint32_t kAliasCount = kAliasView + kElements;
constexpr uint32_t kBeforeGuardCount = 4096 / sizeof(Value);
constexpr uint32_t kAfterGuardCount = 64 / sizeof(Value);
constexpr Bits kSentinel = static_cast<Bits>(0x5a5a5a5au);

template <uint32_t Count> struct Buffer { Value data[Count]; };

constexpr Bits fixture_bits(uint32_t index) {
  if constexpr (sizeof(Bits) == 2)
    return static_cast<Bits>(index * 40503u + 0x35a7u);
  return index * 104729u + 0x80003039u;
}

constexpr Value from_bits(Bits bits) {
  return __builtin_bit_cast(Value, bits);
}

template <uint32_t Count> consteval Buffer<Count> sentinels() {
  Buffer<Count> result{};
  for (uint32_t i = 0; i < Count; ++i)
    result.data[i] = from_bits(kSentinel);
  return result;
}

consteval Buffer<kInputCount> input_fixture() {
  auto result = sentinels<kInputCount>();
  for (uint32_t i = 0; i < kElements; ++i)
    result.data[kInputOffsetElements + i] = from_bits(fixture_bits(i));
  return result;
}

consteval Buffer<kAliasCount> alias_fixture() {
  auto result = sentinels<kAliasCount>();
  for (uint32_t i = 0; i < kElements; ++i)
    result.data[kAliasView + i] = from_bits(fixture_bits(i));
  return result;
}

} // namespace

// The ELF image supplies deterministic shared inputs. Every PE can enter the
// kernel immediately; the fixture introduces no software barrier.
extern "C" {
alignas(4096) Buffer<kBeforeGuardCount> view_copy_input_guard_before =
    sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<kInputCount> view_copy_input = input_fixture();
alignas(sizeof(Value)) Buffer<kAfterGuardCount> view_copy_input_guard_after =
    sentinels<kAfterGuardCount>();
alignas(4096) Buffer<kBeforeGuardCount> view_copy_output_guard_before =
    sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<kOutputCount> view_copy_output =
    sentinels<kOutputCount>();
alignas(sizeof(Value)) Buffer<kAfterGuardCount> view_copy_output_guard_after =
    sentinels<kAfterGuardCount>();
alignas(4096) Buffer<kBeforeGuardCount> view_copy_alias_guard_before =
    sentinels<kBeforeGuardCount>();
alignas(4096) Buffer<kAliasCount> view_copy_alias = alias_fixture();
alignas(sizeof(Value)) Buffer<kAfterGuardCount> view_copy_alias_guard_after =
    sentinels<kAfterGuardCount>();
alignas(32) uint32_t view_copy_seen[4];
alignas(32) uint32_t view_copy_status[16] = {
    VIEW_COPY_PROFILE, kElements, kTileElements,
    kShape[0], kShape[1], kShape[2],
    kInputStride[0], kInputStride[1], kInputStride[2],
    kOutputStride[0], kOutputStride[1], kOutputStride[2],
    kInputOffsetBytes, kOutputOffsetBytes, kAlias, 0x56435031u};
}

int main() {
  const uint32_t pe = get_thread_idx();
  if (pe >= 4)
    return 1;

  uint32_t shape[3] = {kShape[0], kShape[1], kShape[2]};
  uint32_t input_stride[3] = {
      kInputStride[0], kInputStride[1], kInputStride[2]};
  uint32_t output_stride[3] = {
      kOutputStride[0], kOutputStride[1], kOutputStride[2]};

  Value *input = kAlias ? view_copy_alias.data + kAliasInputAdjustment
                        : view_copy_input.data;
  Value *output = kAlias ? view_copy_alias.data : view_copy_output.data;

  BENCHSTART;
  [[clang::noinline]]
  supernpu::tile_isa::tile_view_copy<Value, 3, kElements, kTileElements>(
      input, output, shape, kInputOffsetBytes, kOutputOffsetBytes,
      input_stride, output_stride);
  BENCHEND;

  view_copy_seen[pe] = 0x56430000u | pe;
  return 0;
}
