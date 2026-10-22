#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "basic_op/gather/gather.hpp"
#include <cstddef>
#include <cstdint>

namespace {
template <unsigned Count> struct Buffer { float data[Count]; };
constexpr uint32_t input_bits(unsigned i) {
  constexpr uint32_t edge[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
                               0x7fc12345u, 0x7f800123u, 1, 0x80000001u};
  return (i % 32 < 8) ? edge[i % 32] : (0x3f000000u + i * 8191u);
}
consteval Buffer<8192> input_fixture() {
  Buffer<8192> result{};
  for (unsigned i = 0; i < 8192; ++i)
    result.data[i] = __builtin_bit_cast(float, input_bits(i));
  return result;
}
template <unsigned Count> consteval Buffer<Count> sentinels() {
  Buffer<Count> result{};
  for (unsigned i = 0; i < Count; ++i)
    result.data[i] = __builtin_bit_cast(float, uint32_t{0x5a5a5a5a});
  return result;
}
struct IndexBuffer { int32_t data[128]; };
consteval IndexBuffer index_fixture() {
  IndexBuffer result{};
  for (unsigned i = 0; i < 128; ++i)
    result.data[i] = (i * 37 + 13) % 128;
  return result;
}
} // namespace

// 保留原4PE wrapper的float[128*64]表/输出和int32[128]索引，以及4096字节对齐。
// ELF初始化数据使四个PE共同读取确定输入，无需kernel内barrier。
extern "C" {
alignas(4096) Buffer<8192> gather_fourpe_input = input_fixture();
alignas(4096) IndexBuffer gather_fourpe_indices = index_fixture();
alignas(64) Buffer<896> gather_fourpe_guard_before = sentinels<896>();
alignas(4096) Buffer<8192> gather_fourpe_output = sentinels<8192>();
alignas(64) Buffer<16> gather_fourpe_guard_after = sentinels<16>();
alignas(32) uint32_t gather_fourpe_seen[4];
}

int main() {
  const uint32_t pe = get_thread_idx();
  if (pe >= 4)
    return 1;
  BENCHSTART;
  [[clang::noinline]]
  supernpu::multi_thread::gather<float, int32_t, 128, 128, 64, 32, 64>(
      gather_fourpe_input.data, gather_fourpe_indices.data, gather_fourpe_output.data);
  BENCHEND;
  gather_fourpe_seen[pe] = 0xc04e0000u | pe;
  return 0;
}
