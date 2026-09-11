#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_CUBLAS_FP8 —— 运行期动态 shape 版 driver。kernel 编译期不知 M/K，靠运行期
//   tiling={M,K} 传入（对照 tail_ocp_fp8_dyn driver 范式）。fp16(half) in -> e4m3 out,
//   BlockSize=32, cuBLAS scale。用 fp16 输入规避 bf16 TROWMAX emulator 白名单缺陷2
//   （同静态 tail_cublas_fp8_4pe）。
//
// SPMD：PPE=4 时必须 4 线程跑 gfrun -s softcore.multiThreadNum=4；单线程只写 1/4。
//   RES_CHECK：读 gen 的 fp16 input.bin，写 output.bin + scale_output.bin
//   （compact uint8 E8M0，even-align 补列 scaleCols = evenAlign(PN/PBS)）。
//   无 RES_CHECK 用固定 x=4.0（fp16 0x4400）。
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

    // 动态 shape：M/K 运行期传入（编译期 kernel 不可知）。
    const int64_t tiling[2] = {PM, PN};
    dynamic_mx_quant_tail_cublas_fp8_dyn<PBS, __fp8_e4m3, __half, 0x2b8cbcccu, PPE>(
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
