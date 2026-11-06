#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "solution/transpose/transpose_nd_simt_elementif.hpp"

// element-if（SIMT/SMALL_SHAPE）测试：4D [8,8,8,8] perm(0,2,1,3)
#ifndef SIMT_CASE
#define SIMT_CASE 0
#endif

#if SIMT_CASE == 1
constexpr int kRank = 3;
constexpr int kNumel = 32 * 64 * 512;
#else
constexpr int kRank = 4;
constexpr int kNumel = 8 * 8 * 8 * 8;
#endif

namespace {
alignas(4096) std::int32_t input[kNumel];
alignas(4096) std::int32_t output[kNumel];
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    for (int i = 0; i < kNumel; ++i) input[i] = i;
    for (int i = 0; i < kNumel; ++i) output[i] = -1;
    BENCHSTART;
#if SIMT_CASE == 1
    supernpu::tile_isa::transpose_nd_simt_elementif_tile<
        std::int32_t, 3, kNumel, 512, std::array<int, 8>{32, 64, 512}, std::array<int, 8>{0, 2, 1}>(
        input, output, static_cast<int>(tid), 4, tid == 0);
#else
    supernpu::tile_isa::transpose_nd_simt_elementif_tile<
        std::int32_t, 4, kNumel, 512, std::array<int, 8>{8, 8, 8, 8}, std::array<int, 8>{0, 2, 1, 3}>(
        input, output, static_cast<int>(tid), 4, tid == 0);
#endif
    BENCHEND;
    if (tid == 0) {
        writeBinaryFile(
            "/mnt/workspace/gitCode/ChenYiran/SuperScalar/SuperNPUBench/benchmark/one-level-arch/compare/solution_transpose_transpose_simt_PE4/output.bin",
            reinterpret_cast<std::uint8_t *>(output), sizeof(std::int32_t) * kNumel);
    }
    return 0;
}
