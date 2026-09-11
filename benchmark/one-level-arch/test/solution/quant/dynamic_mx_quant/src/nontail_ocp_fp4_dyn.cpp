#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// NONTAIL_OCP_FP4 —— 运行期动态 shape 版 driver（约束版）。kernel 编译期不知 Axis/Post，
//   靠运行期 tiling={Axis,Post} 传入。fp16(half) in -> fp4_e2m1x2 out, BlockSize=32,
//   TileN=64（编译期模板参）, OCP 值域归约 scale, 非尾轴（归约沿 Axis 行、自由轴 Post 列）。
//
// 支持任意 N（Post 偶数，最右列 tile 动态 ValidCol，见 kernel 头）；前提 Axis % BlockSize == 0。
//   TileN=64（fp4 打包 element-列形 32B 列对齐约束）。
//
// SPMD：kPeNum=4 必须 4 线程跑 gfrun -s softcore.multiThreadNum=4；单线程只写 1/4。
//   RES_CHECK：读 gen（--kernel nontail --algo OCP --dtype FP4 --in-dtype fp16）input.bin，
//   写 output.bin（每行 Post/2 打包字节）+ scale_output.bin。gen 约定 --M=Axis / --K=Post。
//   注：fp4 data 路径基线存在模型侧写侧缺口（SuperScalarModel issues454），output 可能 fail,
//   与 4-PE 切分/动态化无关（切分正确性看 scale 落盘 + tid 分片）。
#ifndef PAXIS
#define PAXIS 512
#endif
#ifndef PPOST
#define PPOST 64
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
static uint8_t y[PAXIS * PPOST / 2] __attribute__((aligned(4096))) = {};  // fp4 packs 2/byte along Post
static uint8_t scale[PSCALE_ROWS * PPOST] __attribute__((aligned(4096))) = {};

#ifndef RES_CHECK
// 编译期覆盖 bf16 / fp32 输入分派分支（仅 build/diss 时实例化，res_check 不跑）。
static __bf16  xbf[32 * 64] __attribute__((aligned(4096))) = {};
static float   xf32[32 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y_bf[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_f32[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_bf[2 * 64] __attribute__((aligned(4096))) = {};
static uint8_t scale_f32[2 * 64] __attribute__((aligned(4096))) = {};
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
    dynamic_mx_quant_nontail_ocp_fp4_dyn<PBS, PTILEN, __fp4_e2m1x2, __half, PPE>(
        x, reinterpret_cast<__fp4_e2m1x2 *>(y), scale, tiling);

#ifndef RES_CHECK
    // Compile-only: exercise the bf16 / fp32 input branches (not res-checked).
    const int64_t tiling_sm[2] = {32, 64};  // Axis=1 block, Post=1 TileN(64)
    dynamic_mx_quant_nontail_ocp_fp4_dyn<PBS, PTILEN, __fp4_e2m1x2, __bf16, 1>(
        xbf, reinterpret_cast<__fp4_e2m1x2 *>(y_bf), scale_bf, tiling_sm);
    dynamic_mx_quant_nontail_ocp_fp4_dyn<PBS, PTILEN, __fp4_e2m1x2, float, 1>(
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
