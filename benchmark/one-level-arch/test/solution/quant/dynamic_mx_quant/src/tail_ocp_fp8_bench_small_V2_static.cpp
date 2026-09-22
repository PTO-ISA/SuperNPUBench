#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp"  // V2：Cube_M32 版
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 bench_small 的 **V2（Cube_M32 布局）静态形状**版：[M=64, N=16384], BS=32,
//   bf16 in -> e4m3 out。与 V1（tail_ocp_fp8_bench_small_V1_static.cpp，RowMajor）**同规格/同数据/
//   同 golden**，唯一区别是 kernel 布局：V2 走 CUBE_M32（VecTileM32 + #311 归约 + TREDUCEPREFIXVIEW）。
//   调静态编译期形状 kernel dynamic_mx_quant_tail_ocp_fp8<M,N,BS,InT>。固定 SPMD 4-PE。
//   依赖 Linx-TileOP-API#180（B.DATR NORM，编译期，PR#191）+ SuperScalarModel#749（gfrun TCVT
//   carrier），均已合入上游 main → 现 gfrun 4-PE 逐字节 PASS，与 V1 一致（曾在依赖未落地时预期 FAIL）。
//   gen 同 V1：--M 64 --K 16384 --block-size 32 --algo OCP --kernel tail --dtype FP8
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
