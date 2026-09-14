#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

static __bf16 x[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale: [M, scaleCols] with scaleCols = evenAlign(K/BlockSize).
// M=8, K=32, BlockSize=32 -> numKb=1 -> scaleCols=2 (col 1 is zero padding).
static uint8_t scale[8 * 2] __attribute__((aligned(4096))) = {};

// Compile-only throwaway buffers for the fp16/fp32 input dispatch branches
// (InT if constexpr). Instantiates the __half and float data paths so the
// disassembly shows all three branches emit; runtime blocked by skew.
static __half xh16[8 * 32] __attribute__((aligned(4096))) = {};
static float  xf32[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_h16[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_f32[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_h16[8 * 2] __attribute__((aligned(4096))) = {};
static uint8_t scale_f32[8 * 2] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_tail_cublas_fp8<8, 32>(x, reinterpret_cast<__fp8_e4m3*>(y), scale);

    // Compile-only: exercise the half / fp32 input branches (not res-checked).
    dynamic_mx_quant_tail_cublas_fp8<8, 32, 32, __fp8_e4m3, __half>(
        xh16, reinterpret_cast<__fp8_e4m3*>(y_h16), scale_h16);
    dynamic_mx_quant_tail_cublas_fp8<8, 32, 32, __fp8_e4m3, float>(
        xf32, reinterpret_cast<__fp8_e4m3*>(y_f32), scale_f32);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
