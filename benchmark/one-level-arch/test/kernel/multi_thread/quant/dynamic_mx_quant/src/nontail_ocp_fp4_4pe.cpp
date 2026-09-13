#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// NONTAIL_OCP_FP4 正式 kernel（固定 SPMD 4-PE）: fp16(half) in -> fp4_e2m1x2 out,
// BlockSize=32, OCP 值域归约 scale, 非尾轴（归约沿 Axis 行、自由轴 Post 列）。计算逻辑
// 与单线程 nontail_ocp_fp4 逐 op 一致，只把外层块行 kb 循环按 get_thread_idx() 切成 4 份
// （kPeNum=4，按块行连续切分，与 nontail_cublas_fp8 同一切分范式）。用 fp16 输入
// （TABS/TCOLMAX 白名单友好）。
//
// SPMD 语义：runtime 把 [0,multiThreadNum) 所有线程 reset 到本 entry，kernel 内靠
//   get_thread_idx() 自我按块行切分。main 仍只调一次。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4；单线程只写 1/4 输出。
//
// RES_CHECK 下读 gen（--kernel nontail --algo OCP --dtype FP4 --in-dtype fp16）的
// input.bin，写 output.bin + scale_output.bin。gen 约定 --M=Axis(归约行) / --K=Post(自由列)。
// 注：非尾轴 scale = compact planar [scaleRows, Post] = PTO-ISA Shared B-scale [G,N]
// 契约（ADR-0101 / pto-spec d0ce06ad）——**无需交织**，scale 与 golden 逐字节精确
// （RECORD 问题5 已解除：规范定义无需交织）。
// 另注：fp4 data 路径当前基线存在模型侧 fp4 写侧缺失（SuperScalarModel issues454），output
// 可能 fail，与 4-PE 切分无关（切分正确性看 scale 落盘与 tid 分片）。
// PPOST=64 → 派生 TileN=64（唯一经原 kernel 验证的 fp4 data 路径宽度；TileN=128 会让
// fp4 打包 TCVT 落到 "ordinary" 路径的 physical-Cols 校验，dst Cols=TileN/2≠src Cols）。
// 4-PE 切分沿 Axis/kb（与 Post 正交），Axis=512/BS=32 → numKb=16 均分 4 PE 已足够。
#ifndef PAXIS
#define PAXIS 512
#endif
#ifndef PPOST
#define PPOST 64
#endif
#define PSCALE_ROWS ((((PAXIS / 32) + 1) / 2) * 2)  // evenAlign(numKb)
static uint16_t xbits[PAXIS * PPOST] __attribute__((aligned(4096))) = {[0 ... PAXIS * PPOST - 1] = 0x3c00};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[PAXIS * PPOST / 2] __attribute__((aligned(4096))) = {};  // fp4 packs 2/byte along Post
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

    dynamic_mx_quant_nontail_ocp_fp4<PAXIS, PPOST, 32, __fp4_e2m1x2, __half, /*kPeNum=*/4>(
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
