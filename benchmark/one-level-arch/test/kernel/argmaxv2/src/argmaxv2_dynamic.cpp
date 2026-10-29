#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/argmaxv2/argmaxv2_dynamic.hpp"

// ArgMaxV2 动态版 fp32 dim=1：[133, 1000] -> int32[133]
// 外层 133 不整除 4PE（ceil 分区 34/34/34/31，替代静态版 PE0 单跑兜底），
// 行尾 34%32=2 与 31%32=31、列尾 1000%16=8 均经运行期 Valid 载体覆盖。
// 输入含 NaN/±inf/平局/±0 等边界模式（由 res_check 预置）。
namespace {
constexpr int kRows = 133;
constexpr int kCols = 1000;
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
    // 运行期 tiling：outer=133, axis=1000, inner=1, 尾轴行归约；
    // tile_m/tile_n = 0 → 物理默认满块（fp32 行归约 [32,16]）。
    supernpu::multi_thread::argmaxv2_dyn::ArgMaxV2TilingData tiling{
        kRows, kCols, 1, 0, 0, 1};
    BENCHSTART;
    supernpu::multi_thread::argmaxv2_dynamic<float, int32_t>(input, output,
                                                             &tiling);
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
