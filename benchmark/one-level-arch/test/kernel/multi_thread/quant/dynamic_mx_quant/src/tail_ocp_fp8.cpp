#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 正式 kernel（固定 SPMD 4-PE）: half(fp16) in -> e4m3 out,
// BlockSize=32, OCP; 倒数用位补（非 TRECIP）。计算逻辑与 single-PE 探针
// probe_ocp_fp8_newcalc 逐 op 一致，只把外层 M 循环按 get_thread_idx() 切成 4 份。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到本 entry，kernel 内靠
//   get_thread_idx() 自我按 M 切分。main 仍只调一次（对照 matmul_shared.cpp）。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4 / gfsim --conf fourpe；
//   单线程跑本 kernel 只会写 1/4 输出。
//
// 接入官方精度流程 (run_precision_check.py: TAIL_OCP_FP8)：RES_CHECK 下读 gen 的
// fp16 input.bin, 写 output.bin + scale_output.bin。scale = compact uint8 E8M0,
// even-align 补列: scaleCols = evenAlign(PN/32)。无 RES_CHECK 时用固定 x=4.0(0x4400)。
#ifndef PM
#define PM 512
#endif
#ifndef PN
#define PN 256
#endif
#define PSCALE_COLS ((((PN / 32) + 1) / 2) * 2)  // even-align block count
static uint16_t xbits[PM * PN] __attribute__((aligned(4096))) = {[0 ... PM * PN - 1] = 0x4400};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[PM * PN] __attribute__((aligned(4096))) = {};
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

    dynamic_mx_quant_tail_ocp_fp8<PM, PN, 32>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
