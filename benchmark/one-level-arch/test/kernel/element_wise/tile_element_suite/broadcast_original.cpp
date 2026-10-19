#include <common/pto_tileop.hpp>
#include "benchmark.h"
#ifdef BROADCAST_INLINE_I32
#include "basic_op/broadcast/broadcast_inline.hpp"
#else
#include "basic_op/broadcast/detail/broadcast.hpp"
#endif
#include <cstddef>
#include <cstdint>

#ifndef BROADCAST_RANK
#define BROADCAST_RANK 2
#endif
#ifndef BROADCAST_ALIAS_PROBE
#define BROADCAST_ALIAS_PROBE 0
#endif

namespace {
#ifdef BROADCAST_INLINE_I32
using Value = int32_t;
using Bits = uint32_t;
constexpr uint32_t kTileElements = 512;
constexpr size_t kInputShape[] = {64, 1, 49};
constexpr size_t kOutputShape[] = {64, 8, 49};
constexpr uint32_t kInputElements = 64 * 49;
constexpr uint32_t kOutputElements = 64 * 8 * 49;
#else
using Value = __half;
using Bits = uint16_t;
constexpr uint32_t kTileElements = 2048;
#if BROADCAST_RANK == 2
constexpr size_t kInputShape[] = {1042, 1};
constexpr size_t kOutputShape[] = {1042, 129};
constexpr uint32_t kInputElements = 1042;
constexpr uint32_t kOutputElements = 1042 * 129;
#elif BROADCAST_RANK == 3
constexpr size_t kInputShape[] = {1280, 1, 49};
constexpr size_t kOutputShape[] = {1280, 8, 49};
constexpr uint32_t kInputElements = 1280 * 49;
constexpr uint32_t kOutputElements = 1280 * 8 * 49;
#elif BROADCAST_RANK == 5
constexpr size_t kInputShape[] = {1, 1, 1, 65, 128};
constexpr size_t kOutputShape[] = {1, 1, 7, 65, 128};
constexpr uint32_t kInputElements = 65 * 128;
constexpr uint32_t kOutputElements = 7 * 65 * 128;
#else
#error unsupported original broadcast rank
#endif
#endif
constexpr bool kAlias = BROADCAST_ALIAS_PROBE != 0;
constexpr Bits kGuard = static_cast<Bits>(0x5a5a5a5a);
constexpr uint32_t kGuardElements = 64 / sizeof(Value);
constexpr uint32_t kInputOffset = 11;
constexpr uint32_t kOutputOffset = 17;
Bits fixture(uint32_t index) {
#ifdef BROADCAST_INLINE_I32
  return index * 104729u + 0x80003039u;
#else
  return static_cast<Bits>(index * 40503u + 0x35a7u);
#endif
}
void set_bits(Value &value, Bits bits) {
  __builtin_memcpy(&value, &bits, sizeof(bits));
}
} // namespace

// 独立 fixture 保留原始 driver 的容量、+11/+17 偏移，以及对应原始 Tile 大小。
// 命名观测区便于 host 逐字节检查；原 driver 的栈分配保持不变。
extern "C" {
alignas(64) Value broadcast_input[kInputElements + kInputOffset];
alignas(64) Value broadcast_output_guard_before[kGuardElements];
alignas(64) Value broadcast_output[kOutputElements + kOutputOffset];
alignas(sizeof(Value)) Value broadcast_output_guard_after[kGuardElements];
alignas(32) uint32_t broadcast_status[8];
}

int main() {
  for (uint32_t i = 0; i < kInputElements + kInputOffset; ++i)
    set_bits(broadcast_input[i], i < kInputOffset ? kGuard : fixture(i - kInputOffset));
  for (uint32_t i = 0; i < kOutputElements + kOutputOffset; ++i)
    set_bits(broadcast_output[i], kGuard);
  for (uint32_t i = 0; i < kGuardElements; ++i) {
    set_bits(broadcast_output_guard_before[i], kGuard);
    set_bits(broadcast_output_guard_after[i], kGuard);
  }
  // 输入位于同一输出分配的+11处，输出仍从+17开始。首 Tile 写入会改变
  // 后续 Tile 的输入；golden 必须按每个 Tile 的读取快照推进。
  if constexpr (kAlias)
    for (uint32_t i = 0; i < kInputElements; ++i)
      set_bits(broadcast_output[kInputOffset + i], fixture(i));
  BENCHSTART;
  [[clang::noinline]]
  broadcast<Value, BROADCAST_RANK, BROADCAST_RANK, BROADCAST_RANK,
            kInputElements, kOutputElements, kTileElements>(
      kAlias ? broadcast_output + kInputOffset : broadcast_input + kInputOffset,
      broadcast_output + kOutputOffset, kInputShape, kOutputShape);
  BENCHEND;
  broadcast_status[0] = kInputElements;
  broadcast_status[1] = kOutputElements;
  broadcast_status[2] = kInputOffset;
  broadcast_status[3] = kOutputOffset;
  broadcast_status[4] = kTileElements;
  broadcast_status[5] = BROADCAST_RANK;
  broadcast_status[6] = kAlias;
  broadcast_status[7] = 0x42524331;
  return 0;
}
