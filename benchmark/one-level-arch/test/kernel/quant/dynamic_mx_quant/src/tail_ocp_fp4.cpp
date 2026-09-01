#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP4 正式 kernel（固定 SPMD 4-PE）: bf16 in -> fp4(e2m1) out（打包 2/字节），
// BlockSize=32, OCP; 倒数用位补（NEWCALC，非 TRECIP，无 TCMPS）。计算结构与 tail_ocp_fp8
// 母本逐级对齐，只把输出换成 fp4 打包、外层 M 循环按 get_thread_idx() 切成 4 份。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到本 entry，kernel 内靠
//   get_thread_idx() 自我按 M 切分。main 仍只调一次。必须用 4 线程跑：
//   gfrun -s softcore.multiThreadNum=4；单线程跑本 kernel 只会写 1/4 输出。
//
// 精度流程（RES_CHECK）：读 gen 的 bf16 input.bin，写 output.bin（每行 PN/2 打包字节）+
//   scale_output.bin（compact uint8 E8M0，even-align 补列 scaleCols = evenAlign(PN/32)）。
//   无 RES_CHECK 时用固定 x=4.0（bf16 0x4080）。
#ifndef PM
#define PM 512
#endif
#ifndef PN
#define PN 256
#endif
#define PSCALE_COLS ((((PN / 32) + 1) / 2) * 2)  // even-align block count
static uint16_t xbits[PM * PN] __attribute__((aligned(4096))) = {[0 ... PM * PN - 1] = 0x4080};
static __bf16  *x = reinterpret_cast<__bf16 *>(xbits);
static uint8_t y[PM * (PN / 2)] __attribute__((aligned(4096))) = {};  // fp4 packed: PN/2 bytes/row
static uint8_t scale[PM * PSCALE_COLS] __attribute__((aligned(4096))) = {};

#ifndef RES_CHECK
// 编译期覆盖 half / fp32 输入分派分支（仅 build/diss 时实例化，res_check 不跑）。
static __half  xh16[8 * 64] __attribute__((aligned(4096))) = {};
static float   xf32[8 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y_h16[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_f32[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_h16[8 * 2] __attribute__((aligned(4096))) = {};
static uint8_t scale_f32[8 * 2] __attribute__((aligned(4096))) = {};
#endif

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
#endif

    dynamic_mx_quant_tail_ocp_fp4<PM, PN, 32>(
        x, reinterpret_cast<__fp4_e2m1x2 *>(y), scale);

#ifndef RES_CHECK
    // Compile-only: exercise the half / fp32 input branches (not res-checked).
    dynamic_mx_quant_tail_ocp_fp4<8, 64, 32, __fp4_e2m1x2, __half>(
        xh16, reinterpret_cast<__fp4_e2m1x2 *>(y_h16), scale_h16);
    dynamic_mx_quant_tail_ocp_fp4<8, 64, 32, __fp4_e2m1x2, float>(
        xf32, reinterpret_cast<__fp4_e2m1x2 *>(y_f32), scale_f32);
#endif

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
