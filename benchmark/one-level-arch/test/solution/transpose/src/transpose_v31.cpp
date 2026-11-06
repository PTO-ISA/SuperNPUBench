#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/transpose/transpose_nd_v3_1.hpp"

// v3.1 场景分派测试（对标 ops-math transpose 的 #1/#5/#7）：
//   CASE=0  #1 TENSOR_MOVE  : 1D [4096], perm=[0]              （dim==1）
//   CASE=1  #5 N_LAST       : 3D [64,64,256], perm=[1,0,2]     （尾轴不转置, 尾轴≥32, bytes≥4MB）
//   CASE=2  #7 GATHER       : 3D [32,64,512], perm=[0,2,1]     （尾轴转置）
#ifndef V31_CASE
#define V31_CASE 0
#endif

#if V31_CASE == 1
constexpr int kRank = 3;
constexpr int kNumel = 64 * 64 * 256;
#elif V31_CASE == 2
constexpr int kRank = 3;
constexpr int kNumel = 32 * 64 * 512;
#elif V31_CASE == 3
constexpr int kRank = 4;
constexpr int kNumel = 8 * 8 * 8 * 8;
#else
constexpr int kRank = 1;
constexpr int kNumel = 4096;
#endif

namespace {
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
#if V31_CASE == 1
    supernpu::tile_isa::transpose_nd_v3_1<std::int32_t, 3, kNumel, 512, 4,
                                          64, 64, 256, 1, 0, 2>(input, output);
#elif V31_CASE == 2
    supernpu::tile_isa::transpose_nd_v3_1<std::int32_t, 3, kNumel, 512, 4,
                                          32, 64, 512, 0, 2, 1>(input, output);
#elif V31_CASE == 3
    supernpu::tile_isa::transpose_nd_v3_1<std::int32_t, 4, kNumel, 512, 4,
                                          8, 8, 8, 8, 0, 2, 1, 3>(input, output);
#else
    supernpu::tile_isa::transpose_nd_v3_1<std::int32_t, 1, kNumel, 512, 4,
                                          4096, 0>(input, output);
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
