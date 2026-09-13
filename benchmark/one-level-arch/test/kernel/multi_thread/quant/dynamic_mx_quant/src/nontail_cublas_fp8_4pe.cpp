#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// NONTAIL_CUBLAS_FP8 正式 kernel（固定 SPMD 4-PE）: fp16(half) in -> e4m3 out,
// BlockSize=32, cuBLAS scale, 非尾轴（归约沿 Axis 行、自由轴 Post 列）。计算逻辑与
// 单线程 nontail_cublas_fp8 逐 op 一致，只把外层块行 kb 循环按 get_thread_idx() 切成
// 4 份（kPeNum=4，按块行连续切分）。用 fp16 输入（TABS/TCOLMAX 白名单友好）。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到本 entry，kernel 内靠
//   get_thread_idx() 自我按块行切分。main 仍只调一次。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4；单线程只写 1/4 输出。
//
// RES_CHECK 下读 gen（--kernel nontail --algo CUBLAS --in-dtype fp16）的 input.bin，
// 写 output.bin + scale_output.bin。gen 约定 --M=Axis(归约行) / --K=Post(自由列)。
// 注：非尾轴 scale = compact planar [scaleRows, Post] = PTO-ISA Shared B-scale [G,N]
// 契约（ADR-0101 / pto-spec d0ce06ad，matmul_shared_lowp.hpp 消费）——**无需交织**。
// scale 与 golden 逐字节精确 pass（RECORD 问题5 已解除：规范定义无需交织）。
#ifndef PAXIS
#define PAXIS 512
#endif
#ifndef PPOST
#define PPOST 256
#endif
#define PSCALE_ROWS ((((PAXIS / 32) + 1) / 2) * 2)  // evenAlign(numKb)
static uint16_t xbits[PAXIS * PPOST] __attribute__((aligned(4096))) = {[0 ... PAXIS * PPOST - 1] = 0x3c00};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[PAXIS * PPOST] __attribute__((aligned(4096))) = {};
static uint8_t scale[PSCALE_ROWS * PPOST] __attribute__((aligned(4096))) = {};

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

    dynamic_mx_quant_nontail_cublas_fp8<PAXIS, PPOST, 32, __fp8_e4m3, __half, 0x2b8cbcccu, /*kPeNum=*/4>(
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
