#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/argmaxv2/argmaxv2.hpp"

// ArgMaxV2 fp32 dim=1：[7, 3333] -> int32[7]（切 R 路径代表例）
// R 大 A 小：ceil(7/4)*4=8 < tM=32 且 R=3333 >= 4*tM*tN/2 → 路由切 R。
// 段几何 seg=832：PE0-2 S=832（16 整除），PE3 S=837（tailB=5）——覆盖
// 末段尾档 + 哨兵 + 屏障合并 + W-big merge 全链。输入含 NaN/±inf/平局/
// ±0 等边界模式（由 res_check 预置）。
namespace {
constexpr int kRows = 7;
constexpr int kCols = 3333;
alignas(4096) float input[kRows * kCols];
alignas(4096) int32_t output[kRows];
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
    supernpu::multi_thread::argmaxv2<float, int32_t, 1, kRows, kCols>(
        input, output);
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
