#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// Non-tail OCP-FP4, LARGE BlockSize (=128) 用例。方案A 切归约轴的 `_bigbs` kernel 已退休；
// 新工具链 tile 上限 256KB 后，单块 [128,64] bf16=16KB 合法，统一 public 入口
// dynamic_mx_quant_nontail_ocp_fp4<Axis,Post,BlockSize> 对大 BS 回退 TileN=align=64 走 plain 单块。
// Axis=128 (=BlockSize, numKb=1), Post=64 (TileN=64=2 MX blocks)。x=[128,64],
// y=[128,32] bytes。scale compact planar uint8 E8M0 [scaleRows, Post]:
// scaleRows = evenAlign(Axis/128) = 2 -> [2,64] bytes。
//
// ★ kPeNum=4：与其余非尾轴用例统一走 4-PE kernel 调用（不再保留单 PE 特例）。
//   numKb = Axis/BlockSize = 1，块行切分下 PE0 处理唯一块行、PE1..3 空转——numKb=1 无块行
//   并行度，故 4-PE 与单 PE 输出逐字节一致。必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
static __bf16 x[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

#ifdef RES_CHECK
static MultiThreadResCheckSync res_check_sync{};  // 4 PE 共享(.bss)：屏障状态
#endif

int main() {
#ifdef RES_CHECK
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
    }
    res_check_publish_inputs(res_check_sync, tid);  // 输入屏障：worker 等 PE0 读完
#endif

    dynamic_mx_quant_nontail_ocp_fp4<128, 64, 128, __fp4_e2m1x2, __bf16, /*kPeNum=*/4>(
        x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
