#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8_newcalc.hpp"
using namespace supernpu::tile_isa::mxquant;

// NEWCALC probe: half(fp16) in -> e4m3 out, BlockSize=32, OCP; 倒数用位补 (非 TRECIP)。
// 接入官方精度流程 (run_precision_check.py: PROBE_OCP_FP8_NEWCALC)：RES_CHECK 下读 gen 的
// fp16 input.bin, 写 output.bin + scale_output.bin。scale = compact uint8 E8M0,
// even-align 补列: scaleCols = evenAlign(PN/32)。无 RES_CHECK 时用固定 x=4.0(0x4400)。
#ifndef PM
#define PM 8
#endif
#ifndef PN
#define PN 32
#endif
#define PSCALE_COLS ((((PN / 32) + 1) / 2) * 2)  // even-align block count
static uint16_t xbits[PM * PN] __attribute__((aligned(4096))) = {[0 ... PM * PN - 1] = 0x4400};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[PM * PN] __attribute__((aligned(4096))) = {};
static uint8_t scale[PM * PSCALE_COLS] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)xbits, sizeof(xbits));
#endif

    probe_dynamic_mx_quant_tail_ocp_fp8_newcalc<PM, PN, 32>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
