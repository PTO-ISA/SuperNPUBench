#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8_dyn_V1-09181200.hpp"  // V1：RowMajor dyn
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 bench_small 的 **V1（RowMajor 布局）动态 shape** 版：[M=64, N=16384], BS=32,
//   bf16 in -> e4m3 out。与 V2（tail_ocp_fp8_bench_small_V2_dyn.cpp，Cube_M32 dyn）同规格/同数据/
//   同 golden，仅 kernel 布局不同：V1 走 RowMajor（Tile<>/TileM=128 + #585 列切归约），M/N 运行期传入。
//   调动态 shape kernel dynamic_mx_quant_tail_ocp_fp8_dyn<BS,PE,InT>（此处解析到 V1 备份头里的
//   RowMajor 实现）。V1 不依赖任何在研 issue，官方基线即 PASS，可进 compile.all 喂 gfsim。
//   与静态 V1（tail_ocp_fp8_bench_small_V1_static.cpp）算法一致，仅 shape 编译期 vs 运行期之别。
//   gen 同 V2：--M 64 --K 16384 --block-size 32 --algo OCP --kernel tail --dtype FP8
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

    const int64_t tiling[2] = {PM, PN};   // 动态 shape：M/N 运行期传入
    dynamic_mx_quant_tail_ocp_fp8_dyn<32, 4, __bf16>(
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
