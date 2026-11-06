#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/transpose/transpose_nd_v4.hpp"

// v4（对标 ops-math transpose_v2：固定 perm）测试。
//   PERM_CASE=0: Per0213  [B,N,S,D]=[1,16,936,128] → [1,936,16,128]  (BNSD→BSND)
//   PERM_CASE=1: Per021   [A,B,C]=[2,128,512]      → [2,512,128]
//   PERM_CASE=2: Per102   [A,B,C]=[16,32,64]       → [32,16,64]
#ifndef TRANSPOSE_TILE_ELEMENTS
#define TRANSPOSE_TILE_ELEMENTS 512
#endif
#ifndef PERM_CASE
#define PERM_CASE 0
#endif

namespace {
#if PERM_CASE == 1
constexpr int kD0 = 2, kD1 = 128, kD2 = 512, kD3 = 1;
constexpr int kNumel = kD0 * kD1 * kD2;
#elif PERM_CASE == 2
constexpr int kD0 = 16, kD1 = 32, kD2 = 64, kD3 = 1;
constexpr int kNumel = kD0 * kD1 * kD2;
#else
constexpr int kD0 = 1, kD1 = 16, kD2 = 936, kD3 = 128;
constexpr int kNumel = kD0 * kD1 * kD2 * kD3;
#endif
alignas(4096) std::int32_t input[kNumel];
alignas(4096) std::int32_t output[kNumel];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
#if PERM_CASE == 1
    supernpu::multi_thread::transpose_nd_v4<
        supernpu::tile_isa::TransposeV2Perm::Per021, std::int32_t, kD0, kD1, kD2, kD3, TRANSPOSE_TILE_ELEMENTS>(
        input, output);
#elif PERM_CASE == 2
    supernpu::multi_thread::transpose_nd_v4<
        supernpu::tile_isa::TransposeV2Perm::Per102, std::int32_t, kD0, kD1, kD2, kD3, TRANSPOSE_TILE_ELEMENTS>(
        input, output);
#else
    supernpu::multi_thread::transpose_nd_v4<
        supernpu::tile_isa::TransposeV2Perm::Per0213, std::int32_t, kD0, kD1, kD2, kD3, TRANSPOSE_TILE_ELEMENTS>(
        input, output);
#endif
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }
#endif
    return 0;
}
