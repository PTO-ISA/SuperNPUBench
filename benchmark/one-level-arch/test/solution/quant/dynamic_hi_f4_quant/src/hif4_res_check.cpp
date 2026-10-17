#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "solution/quant/dynamic_hi_f4_quant/dynamic_hi_f4_quant_tail.h"
using namespace supernpu::tile_isa::hif4quant;

// ============================================================================
// dynamic_hi_f4_quant 数值验证 driver（单-PE）
// ============================================================================
// kernel 非 SPMD（单线程循环全 M），故无 4-PE barrier：跑 multiThreadNum=1，PE0 读
// input.bin → kernel → 写 output.bin(fp4 packed, PM*PN/2 字节) + scale_output.bin
// (U32/块, PM*(PN/64)*4 字节)。golden/compare 见 gen_hif4_golden.py / hif4_compare.py。
// 无 RES_CHECK 时用零输入（保留 emit 见证，等价 FRAMEWORK）。
// 接入 run_hif4_precision.py：RES_CHECK 下 CHK_DIR 由 Makefile res_check=on 注入。
#ifndef PM
#define PM 32
#endif
#ifndef PN
#define PN 128
#endif

static uint16_t xbits[PM * PN]        __attribute__((aligned(4096))) = {};
static __bf16  *x = reinterpret_cast<__bf16 *>(xbits);
static uint8_t  y[PM * PN / 2]        __attribute__((aligned(4096))) = {};
static uint32_t scale[PM * (PN / 64)] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)xbits, sizeof(xbits));
#endif

    dynamic_hi_f4_quant_tail<PM, PN, 64, __fp4_e1m2x2, __bf16>(
        x, reinterpret_cast<__fp4_e1m2x2 *>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin",       (uint8_t *)y,     sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t *)scale, sizeof(scale));
#endif
    return 0;
}
