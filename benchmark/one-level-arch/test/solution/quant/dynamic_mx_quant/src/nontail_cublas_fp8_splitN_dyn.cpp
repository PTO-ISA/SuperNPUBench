#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8_splitN_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// NONTAIL_CUBLAS_FP8 SPLIT-N —— 运行期动态 shape，**按自由轴 N（列 tile）切分 4-PE** 版 driver。
//   计算与 nontail_cublas_fp8_dyn 逐 op 一致，仅 PE 切分维度改为列 tile 索引 n（TILE 粒度）。
//   fp16(half) in -> e4m3 out, BlockSize=32, TileN=32, cuBLAS scale。
//   支持任意 N（见 kernel 头）；前提 Axis%BlockSize==0。PPOST=256/TileN=32 → numN=8，4 PE 各 2 列 tile。
//   RES_CHECK：读 gen（--kernel nontail --algo CUBLAS --in-dtype fp16）input.bin，写
//   output.bin + scale_output.bin。gen 约定 --M=Axis / --K=Post。
#ifndef PAXIS
#define PAXIS 512
#endif
#ifndef PPOST
#define PPOST 256
#endif
#ifndef PBS
#define PBS 32
#endif
#ifndef PTILEN
#define PTILEN 128
#endif
#ifndef PPE
#define PPE 4
#endif
#define PSCALE_ROWS ((((PAXIS / PBS) + 1) / 2) * 2)  // evenAlign(numKb)
static uint16_t xbits[PAXIS * PPOST] __attribute__((aligned(4096))) = {[0 ... PAXIS * PPOST - 1] = 0x3c00};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[PAXIS * PPOST] __attribute__((aligned(4096))) = {};
static uint8_t scale[PSCALE_ROWS * PPOST] __attribute__((aligned(4096))) = {};

#ifndef RES_CHECK
// 编译期覆盖 bf16 / fp32 输入分派分支（仅 build/diss 时实例化）。
static __bf16  xbf[32 * 32] __attribute__((aligned(4096))) = {};
static float   xf32[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_bf[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_f32[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_bf[2 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_f32[2 * 32] __attribute__((aligned(4096))) = {};
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

    // 动态 shape：Axis/Post 运行期传入。
    const int64_t tiling[2] = {PAXIS, PPOST};
    dynamic_mx_quant_nontail_cublas_fp8_splitN_dyn<PBS, PTILEN, __fp8_e4m3, __half, 0x2b8cbcccu, PPE>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale, tiling);

#ifndef RES_CHECK
    const int64_t tiling_sm[2] = {32, 32};  // Axis=1 block, Post=1 TileN(32)
    dynamic_mx_quant_nontail_cublas_fp8_splitN_dyn<PBS, PTILEN, __fp8_e4m3, __bf16, 0x2b8cbcccu, 1>(
        xbf, reinterpret_cast<__fp8_e4m3 *>(y_bf), scale_bf, tiling_sm);
    dynamic_mx_quant_nontail_cublas_fp8_splitN_dyn<PBS, PTILEN, __fp8_e4m3, float, 0x2b8cbcccu, 1>(
        xf32, reinterpret_cast<__fp8_e4m3 *>(y_f32), scale_f32, tiling_sm);
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
