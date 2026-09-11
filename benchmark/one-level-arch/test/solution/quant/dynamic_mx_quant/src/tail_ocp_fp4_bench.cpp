#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP4 大 shape 基准用例：[M=15360, N=1536], BlockSize=32, bf16 in -> fp4(e2m1) out。
//   与 tail_ocp_fp4.cpp 同一 kernel/算法，仅换成真实规模的固定形状（专用 driver = 专用 .o，
//   避免与 512×256 用例共享 tail_ocp_fp4.o 造成 -DPM 陈旧复用）。固定 SPMD 4-PE：M 按 tid
//   切 4 份，每 PE 3840 行 = 30×TileM(128)，整除无 boxed 尾块。numKb = 1536/32 = 48。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
//   RES_CHECK：读 gen（--M 15360 --K 1536 --block-size 32 --algo OCP --kernel tail --dtype FP4
//   --in-dtype bf16 --scale-layout compact）的 input.bin，写 output.bin（每行 N/2 打包字节）+
//   scale_output.bin（compact uint8 E8M0，scaleCols = evenAlign(N/32) = 48）。
#ifndef PM
#define PM 15360
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

    dynamic_mx_quant_tail_ocp_fp4<PM, PN, 32>(
        x, reinterpret_cast<__fp4_e2m1x2 *>(y), scale);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
