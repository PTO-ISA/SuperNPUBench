#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "spmd_res_check.h"  // 4-PE 收尾协议：leader-gate I/O + 两道屏障（issue #489）
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
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

int main() {
#ifdef RES_CHECK
    SpmdResCheck<4> io;  // leader(tid0) 读输入，worker 自旋等齐（输入屏障）
    io.leader_load(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
#endif

    dynamic_mx_quant_nontail_cublas_fp8<PAXIS, PPOST, 32, __fp8_e4m3, __half, 0x2b8cbcccu, /*kPeNum=*/4>(
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
