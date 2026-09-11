#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP4 —— 运行期动态 shape 版 driver。kernel 编译期不知 M/N，靠运行期
//   tiling={M,N} 传入（对照 tail_ocp_fp8_dyn driver 范式）。BlockSize/OutT/InT/kPeNum
//   为属性 → 模板参。host buffer 仍按 PM/PN 静态开（host 必须知尺寸），传给 kernel 的
//   shape 是运行期 int64。
//
// SPMD：PPE=4 时必须 4 线程跑 gfrun -s softcore.multiThreadNum=4；单线程只写 1/4。
//   RES_CHECK：读 gen 的 bf16 input.bin，写 output.bin（每行 PN/2 打包字节）+
//   scale_output.bin（compact uint8 E8M0，even-align 补列 scaleCols = evenAlign(PN/PBS)）。
//   无 RES_CHECK 用固定 x=4.0（bf16 0x4080）。
#ifndef PM
#define PM 512
#endif
#ifndef PN
#define PN 256
#endif
#ifndef PBS
#define PBS 32
#endif
#ifndef PPE
#define PPE 4
#endif
#define PSCALE_COLS ((((PN / PBS) + 1) / 2) * 2)  // even-align block count
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

    // 动态 shape：M/N 运行期传入（编译期 kernel 不可知）。
    const int64_t tiling[2] = {PM, PN};
    dynamic_mx_quant_tail_ocp_fp4_dyn<PBS, __fp4_e2m1x2, __bf16, PPE>(
        x, reinterpret_cast<__fp4_e2m1x2 *>(y), scale, tiling);

#ifndef RES_CHECK
    // Compile-only: exercise the half / fp32 input branches (not res-checked).
    const int64_t tiling_sm[2] = {8, 64};
    dynamic_mx_quant_tail_ocp_fp4_dyn<PBS, __fp4_e2m1x2, __half, 1>(
        xh16, reinterpret_cast<__fp4_e2m1x2 *>(y_h16), scale_h16, tiling_sm);
    dynamic_mx_quant_tail_ocp_fp4_dyn<PBS, __fp4_e2m1x2, float, 1>(
        xf32, reinterpret_cast<__fp4_e2m1x2 *>(y_f32), scale_f32, tiling_sm);
#endif

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
