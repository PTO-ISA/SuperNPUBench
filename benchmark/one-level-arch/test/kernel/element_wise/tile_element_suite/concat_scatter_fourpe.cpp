#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "basic_op/concat/concat_scatter.hpp"
#include <cstddef>
#include <cstdint>

namespace {
template <unsigned Count> struct Buffer { float data[Count]; };
constexpr uint32_t input_bits(unsigned i) {
  constexpr uint32_t edge[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
                               0x7fc12345u, 0x7f800123u, 1, 0x80000001u};
  return (i % 32 < 8) ? edge[i % 32] : (0x3f000000u + i * 8191u);
}
consteval Buffer<512> input_fixture() {
  Buffer<512> result{};
  for (unsigned i = 0; i < 512; ++i)
    result.data[i] = __builtin_bit_cast(float, input_bits(i));
  return result;
}
template <unsigned Count> consteval Buffer<Count> sentinels() {
  Buffer<Count> result{};
  for (unsigned i = 0; i < Count; ++i)
    result.data[i] = __builtin_bit_cast(float, uint32_t{0x5a5a5a5a});
  return result;
}
} // namespace

// 保留原4PE wrapper的float[512]/float[1024]和4096字节对齐。
// ELF初始化数据使四个PE共同读取确定输入，无需kernel内barrier。
extern "C" {
alignas(4096) Buffer<512> concat_scatter_fourpe_input = input_fixture();
alignas(64) Buffer<512> concat_scatter_fourpe_guard_before = sentinels<512>();
alignas(4096) Buffer<1024> concat_scatter_fourpe_output = sentinels<1024>();
alignas(64) Buffer<16> concat_scatter_fourpe_guard_after = sentinels<16>();
alignas(32) uint32_t concat_scatter_fourpe_seen[4];
}

int main() {
  const uint32_t pe = get_thread_idx();
  if (pe >= 4)
    return 1;
  size_t input_shape[2] = {32, 16};
  size_t output_shape[2] = {32, 32};
  BENCHSTART;
  [[clang::noinline]]
  supernpu::multi_thread::concat_scatter<float, 8, 512, 1024, 128, 2, 1>(
      concat_scatter_fourpe_input.data, concat_scatter_fourpe_output.data,
      input_shape, output_shape);
  BENCHEND;
  concat_scatter_fourpe_seen[pe] = 0xc04e0000u | pe;
  return 0;
}
