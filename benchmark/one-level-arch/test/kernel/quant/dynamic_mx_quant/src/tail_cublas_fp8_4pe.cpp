#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "spmd_res_check.h"  // 4-PE 收尾协议：leader-gate I/O + 两道屏障（issue #489）
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_CUBLAS_FP8 正式 kernel（固定 SPMD 4-PE）: fp16(half) in -> e4m3 out,
// BlockSize=32, cuBLAS scale。计算逻辑与单线程 tail_cublas_fp8 逐 op 一致，只把外层
// M 循环按 get_thread_idx() 切成 4 份（kPeNum=4）。用 fp16 输入规避 bf16 TROWMAX
// emulator 白名单缺陷2（同 tail_ocp_fp8）。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到本 entry，kernel 内靠
//   get_thread_idx() 自我按 M 切分。main 仍只调一次。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4；单线程只写 1/4 输出。
//
// RES_CHECK 下读 gen 的 fp16 input.bin，写 output.bin + scale_output.bin。
// scale = compact uint8 E8M0，even-align 补列：scaleCols = evenAlign(PN/32)。
// 无 RES_CHECK 时用固定 x=4.0(fp16 0x4400)。
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

int main() {
#ifdef RES_CHECK
    SpmdResCheck<4> io;  // leader(tid0) 读输入，worker 自旋等齐（输入屏障）
    io.leader_load(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
#endif

    dynamic_mx_quant_tail_cublas_fp8<PM, PN, 32, __fp8_e4m3, __half, 0x2b8cbcccu, /*kPeNum=*/4>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);

#ifdef RES_CHECK
    // 仅 leader(tid0) 落盘（消除 O_TRUNC 竞争）；完整性由 leader-heaviest 块级锁步保证，无需
    // 输出屏障；worker 返回后 runtime park（详见 spmd_res_check.h / issue 489）。
    if (io.leader_should_write()) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
