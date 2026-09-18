#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp"  // 静态 M32 版
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 bench_small 的**静态形状**版：[M=64, N=16384], BlockSize=32, bf16 in -> e4m3 out。
//   与 tail_ocp_fp8_bench_small.cpp 同规格/同数据，唯一区别：调**静态编译期形状** kernel
//   dynamic_mx_quant_tail_ocp_fp8<M,N,BS,InT>（编译期 M/N/ValidRows），供 M32 + TREDUCEPREFIXVIEW
//   范式测试（该范式的 asm 要编译期 ValidRow，运行期 _dyn 用不了）。固定 SPMD 4-PE。
//   gen 同 bench_small：--M 64 --K 16384 --block-size 32 --algo OCP --kernel tail --dtype FP8
//   --in-dtype bf16 --scale-layout compact。
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

    // 静态编译期形状 kernel（无 tiling 指针）。固定 4-PE（kernel 内部 kPeNum=4）。
    dynamic_mx_quant_tail_ocp_fp8<PM, PN, 32, __bf16>(
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
