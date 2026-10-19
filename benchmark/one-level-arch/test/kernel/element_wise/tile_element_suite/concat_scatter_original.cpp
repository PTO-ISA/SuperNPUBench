#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "linx_print.h"
#include "basic_op/concat/detail/concat_scatter.hpp"
#include <cstddef>
#include <cstdint>

#ifndef CONCAT_SCATTER_FP16
#define CONCAT_SCATTER_FP16 0
#endif
#ifndef CONCAT_SCATTER_ALIAS_PROBE
#define CONCAT_SCATTER_ALIAS_PROBE 0
#endif

namespace {
#if CONCAT_SCATTER_FP16
using Value = __half;
using Bits = uint16_t;
constexpr uint32_t kInputElements = 8192000;
constexpr uint32_t kOutputElements = 2048000;
constexpr uint32_t kRows = 256;
constexpr uint32_t kColumns = 8;
#else
using Value = int32_t;
using Bits = uint32_t;
constexpr uint32_t kInputElements = 128000;
constexpr uint32_t kOutputElements = 128000;
constexpr uint32_t kRows = 64;
constexpr uint32_t kColumns = 2;
#endif
constexpr bool kAlias = CONCAT_SCATTER_ALIAS_PROBE != 0;
constexpr uint32_t kProcessed = kAlias ? 1024 : kOutputElements;
constexpr uint32_t kAliasOffset = 509;
constexpr uint32_t kGuardElements = 64 / sizeof(Value);
constexpr Bits kGuard = static_cast<Bits>(0x5a5a5a5au);

Bits fixture(uint32_t index) {
#if CONCAT_SCATTER_FP16
  return static_cast<Bits>(index * 40503u + 0x35a7u);
#else
  return index * 104729u + 0x80003039u;
#endif
}
void set_bits(Value &value, Bits bits) {
  __builtin_memcpy(&value, &bits, sizeof(bits));
}
Bits get_bits(const Value &value) {
  Bits bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return bits;
}
} // namespace

// 原 driver 的输入未初始化、文件读取关闭。本 fixture 保留完整分配，
// 使用确定性原始位；FP16 包含非有限值、subnormal 与正负零。
#ifdef CONCAT_SCATTER_F16_LINKED_INPUTS
static_assert(CONCAT_SCATTER_FP16 == 1);
#define CSF16_DATA_DECL extern
#else
#define CSF16_DATA_DECL
#endif
extern "C" {
alignas(4096) CSF16_DATA_DECL Value concat_scatter_input[kInputElements];
alignas(64) CSF16_DATA_DECL Value concat_scatter_output_guard_before[kGuardElements];
alignas(64) CSF16_DATA_DECL Value concat_scatter_output[kOutputElements];
alignas(64) CSF16_DATA_DECL Value concat_scatter_output_guard_after[kGuardElements];
alignas(32) CSF16_DATA_DECL size_t concat_scatter_in_shape[8];
alignas(32) CSF16_DATA_DECL size_t concat_scatter_out_shape[8];
alignas(32) uint32_t concat_scatter_status[12];
}
#undef CSF16_DATA_DECL
#ifdef CONCAT_SCATTER_F16_LINKED_INPUTS
#include "concat_scatter_f16_input.inc"
#endif

int main() {
#ifndef CONCAT_SCATTER_F16_LINKED_INPUTS
  for (uint32_t i = 0; i < kInputElements; ++i)
    set_bits(concat_scatter_input[i], fixture(i));
  for (uint32_t i = 0; i < kOutputElements; ++i)
    set_bits(concat_scatter_output[i], kGuard);
  for (uint32_t i = 0; i < kGuardElements; ++i) {
    set_bits(concat_scatter_output_guard_before[i], kGuard);
    set_bits(concat_scatter_output_guard_after[i], kGuard);
  }
  concat_scatter_in_shape[0] = kRows;
  concat_scatter_in_shape[1] = kColumns;
  concat_scatter_out_shape[0] = kRows;
  concat_scatter_out_shape[1] = 1000 * kColumns;
#endif

  BENCHSTART;
  // alias 探针保留完整容量，连续处理两个512元素批次。第一个批次会覆盖
  // 下一批次的输入；每个批次必须先完整快照，再执行本批次的任何写入。
  [[clang::noinline]]
  concat_scatter<Value, 8, kInputElements, kOutputElements, 512, 2, 1>(
      concat_scatter_input,
      kAlias ? concat_scatter_input + kAliasOffset : concat_scatter_output,
      concat_scatter_in_shape, concat_scatter_out_shape, 0, kProcessed);
  BENCHEND;

  uint32_t guard_failures = 0;
  for (uint32_t i = 0; i < kGuardElements; ++i) {
    guard_failures += get_bits(concat_scatter_output_guard_before[i]) != kGuard;
    guard_failures += get_bits(concat_scatter_output_guard_after[i]) != kGuard;
  }
  concat_scatter_status[0] = guard_failures;
  concat_scatter_status[1] = kInputElements;
  concat_scatter_status[2] = kOutputElements;
  concat_scatter_status[3] = kProcessed;
  concat_scatter_status[4] = 512;
  concat_scatter_status[5] = kRows;
  concat_scatter_status[6] = kColumns;
  concat_scatter_status[7] = 1000 * kColumns;
  concat_scatter_status[8] = 1000;
  concat_scatter_status[9] = kAlias;
  concat_scatter_status[10] = kAlias ? kAliasOffset : 0;
  concat_scatter_status[11] = 0x43534331;
  // 完整输入、输出和未使用尾部由独立 host golden 检查，不嵌入kernel答案。
  return guard_failures != 0;
}
