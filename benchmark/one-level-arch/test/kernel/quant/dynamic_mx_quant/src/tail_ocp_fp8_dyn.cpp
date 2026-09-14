#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8_dyn.hpp"
using namespace supernpu::tile_isa::mxquant;

// TAIL_OCP_FP8 —— 运行期动态 shape 版 driver。
//   kernel 编译期不知 M/N，靠运行期 tiling={M,N} 传入（对照 rms_norm driver 范式）。
//   BlockSize、kPeNum 为属性 → 模板参。host buffer 仍按 PM/PN 静态开（host 必须知尺寸），
//   但传给 kernel 的 shape 是运行期 int64。
//
// SPMD：PPE=4 时必须 4 线程跑 gfrun -s softcore.multiThreadNum=4；单线程只写 1/4。
//   接入精度流程（RES_CHECK）：读 fp16 input.bin，写 output.bin + scale_output.bin。
//   scale = compact uint8 E8M0，even-align 补列 scaleCols = evenAlign(PN/PBS)。
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

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
#endif

    // 动态 shape：M/N 运行期传入（编译期 kernel 不可知）。
    const int64_t tiling[2] = {PM, PN};
    dynamic_mx_quant_tail_ocp_fp8_dyn<PBS, PPE>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale, tiling);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
