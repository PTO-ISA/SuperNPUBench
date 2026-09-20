#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8_dyn.hpp"  // V2：Cube_M32 dyn
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 bench_small 的 **V2（Cube_M32 布局）动态 shape** 版：[M=64, N=16384], BS=32,
//   bf16 in -> e4m3 out。与 V1（tail_ocp_fp8_bench_small_V1_dyn.cpp，RowMajor dyn）同规格/同数据/
//   同 golden，仅 kernel 布局不同：V2 走 CUBE_M32（VecTileM32 + #311 归约 + TREDUCEPREFIXVIEW）。
//   ⚠ 已知**编译阻塞**（Linx-TileOP-API 缺口，非 ISA/规范限制，可修）：M32 的 #311 归约结果只能靠
//   TREDUCEPREFIXVIEW 消费；而 TileOP 头的 reduction-prefix 发射路径（pto_tile_region_inline_asm.hpp）
//   只实现了**立即数维度** `B.DIM zero, %c(ValidRow)`（并 static_assert(ValidRow>0)），缺 TCVT 头已有的
//   **动态分支**（ValidRow<0 → `B.DIM %[reg],0` + `"r"(src.GetValidRow())`）。动态 shape 传 ValidRow=-1
//   塞进无符号 uimm → 发出 `B.DIM zero, -1` → `Match Instruction Error`。ISA 层 B.DIM 本支持寄存器维度
//   （V1_dyn / TCVT 都在用），故此为 TileOP 头 reduction-prefix 未接动态维度的缺口，补上该分支即可。
//   在补齐前，M32 + reduction-prefix 只能做**静态** kernel（见 tail_ocp_fp8_bench_small_V2_static.cpp）。
//   本 V2_dyn 作为该 TileOP 缺口的 witness 保留，**当前预期编译失败**；不进 compile.all。
//   固定 SPMD 4-PE：M 按 tid 切 4 份，每 PE 16 行。numKb = 16384/32 = 512。必须 4 线程跑：
//   gfrun -s softcore.multiThreadNum=4。
//   RES_CHECK：读 gen（--M 64 --K 16384 --block-size 32 --algo OCP --kernel tail --dtype FP8
//   --in-dtype bf16 --scale-layout compact）的 input.bin，写 output.bin（1 字节/元素 e4m3）+
//   scale_output.bin（compact uint8 E8M0，scaleCols = evenAlign(N/32) = 512）。
#ifndef PM
#define PM 64
#endif
#ifndef PN
#define PN 16384
#endif
#define PSCALE_COLS ((((PN / 32) + 1) / 2) * 2)  // even-align block count = 512
static uint16_t xbits[PM * PN] __attribute__((aligned(4096))) = {[0 ... PM * PN - 1] = 0x4080};
static __bf16  *x = reinterpret_cast<__bf16 *>(xbits);
static uint8_t y[PM * PN] __attribute__((aligned(4096))) = {};             // e4m3: 1 字节/元素
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
    dynamic_mx_quant_tail_ocp_fp8_dyn<32, 4, __bf16>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale, tiling);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
