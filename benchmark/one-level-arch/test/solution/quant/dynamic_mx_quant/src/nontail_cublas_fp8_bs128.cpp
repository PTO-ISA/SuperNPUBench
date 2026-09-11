#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread_res_check.h"  // 官方 4-PE 收尾协议（输入/输出屏障 + PE0 落盘）
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// End-to-end RES_CHECK harness for LARGE BlockSize (=128) non-tail cuBLAS-FP8。
// 方案A 切归约轴的 `_bigbs` kernel 已退休；统一走 public 入口
// dynamic_mx_quant_nontail_cublas_fp8<Axis, Post, BlockSize>，整块 [128,32] 单 tile
// 载入（32b 中间量 16KB，在 256KB TilesizeCode 上限内），TileN 回退 align=32。
//
// ★ kPeNum=4：与其余非尾轴用例统一走 4-PE kernel 调用（不再保留单 PE 特例）。
//   BlockSize=128 → numKb = Axis/BlockSize = 1，块行切分下 PE0 处理唯一块行、PE1..3
//   空转（kb_begin==kb_end）——numKb=1 无块行并行度，故 4-PE 与单 PE 输出逐字节一致。
//   必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
//
// nontail: reduce along rows (Axis), Post is the free column axis。
//   Axis=128 (=1 reduce block, BlockSize=128), Post=32。
//   scaleRows = evenAlign(Axis/BlockSize) = evenAlign(1) = 2 -> scale[2, 32]。
// golden: gen_dynamic_mx_quant_data.py --M 128 --K 32 --block-size 128
//         --algo CUBLAS --kernel nontail --dtype FP8 --scale-layout compact --in-dtype bf16
static __bf16  x[128 * 32]     __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32]     __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 32]   __attribute__((aligned(4096))) = {};

#ifdef RES_CHECK
static MultiThreadResCheckSync res_check_sync{};  // 4 PE 共享(.bss)：屏障状态
#endif

int main() {
#ifdef RES_CHECK
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
    }
    res_check_publish_inputs(res_check_sync, tid);  // 输入屏障：worker 等 PE0 读完
#endif

    // Axis=128, Post=32, BlockSize=128 -> single-load plain (TileN=align=32)，kPeNum=4。
    dynamic_mx_quant_nontail_cublas_fp8<128, 32, 128, __fp8_e4m3, __bf16, 0x2b8cbcccu, /*kPeNum=*/4>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);  // 输出屏障：PE0 等 PE1..3 算完再落盘
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
        writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
    }
#endif
    return 0;
}
