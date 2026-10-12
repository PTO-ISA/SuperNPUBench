#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP4 缩小版网络基准用例：[M=256, N=1536], BlockSize=32, bf16 in -> fp4(e2m1) out。
//   与 tail_ocp_fp4_bench.cpp 完全同一 kernel/算法/结构，仅把默认形状从 [15360,1536] 缩到
//   [256,1536]（专用 driver = 专用 .o，避免与其它用例共享 .o 造成 -DPM 陈旧复用）。缩小版
//   足够小（256×1536 bf16≈0.75MB），可进 compile.all 默认跑、直接喂 gfsim 做网络形状时序。
//   固定 SPMD 4-PE：M 按 tid 切 4 份，每 PE 64 行 = boxed 尾块（64 < TileM=128，seg_full=0）。
//   numKb = 1536/32 = 48。必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
//   RES_CHECK：读 gen（--M 256 --K 1536 --block-size 32 --algo OCP --kernel tail --dtype FP4
//   --in-dtype bf16 --scale-layout compact）的 input.bin，写 output.bin（每行 N/2 打包字节）+
//   scale_output.bin（compact uint8 E8M0，scaleCols = evenAlign(N/32) = 48）。
#ifndef PM
#define PM 256
#endif
#ifndef PN
#define PN 1536
#endif
#define PSCALE_COLS ((((PN / 32) + 1) / 2) * 2)  // even-align block count = 48
static uint16_t xbits[PM * PN] __attribute__((aligned(4096))) = {[0 ... PM * PN - 1] = 0x4080};
static __bf16  *x = reinterpret_cast<__bf16 *>(xbits);
static uint8_t y[PM * (PN / 2)] __attribute__((aligned(4096))) = {};  // fp4 packed: PN/2 bytes/row
static uint8_t scale[PM * PSCALE_COLS] __attribute__((aligned(4096))) = {};

#ifdef RES_CHECK
static MultiThreadResCheckSync res_check_sync{};  // 4 PE 共享(.bss)：屏障状态
#endif

int main() {
#ifdef RES_CHECK
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
    }
    res_check_publish_inputs(res_check_sync, tid);  // 输入屏障：worker 等 PE0 读完
#endif

    const int64_t tiling[2] = {PM, PN};   // 动态 shape：M/N 运行期传入
    dynamic_mx_quant_tail_ocp_fp4_dyn<32, __fp4_e2m1x2, __bf16, 4>(
        x, reinterpret_cast<__fp4_e2m1x2 *>(y), scale, tiling);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
