#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// FP4 output packed 2/byte -> y holds K/2 bytes per row. Tail fp4 packs 2 MX
// blocks per output tile (a single 32-value block = 16B fails 32B alignment; see
// RECORD.md 问题2), so K must be >= 2*BlockSize. K=64 = 2 blocks: x=[8,64],
// y=[8,32] bytes. scale is compact planar uint8 E8M0 [M, scaleCols] with
// scaleCols = evenAlign(K/32) = 2 -> [8,2] bytes.
static __bf16 x[8 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[8 * 2] __attribute__((aligned(4096))) = {};

// Compile-only throwaway buffers for the fp16/fp32 input dispatch branches
// (InT if constexpr). Instantiates the half / fp32 OCP regularizers + data paths.
static __half xh16[8 * 64] __attribute__((aligned(4096))) = {};
static float  xf32[8 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y_h16[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_f32[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_h16[8 * 2] __attribute__((aligned(4096))) = {};
static uint8_t scale_f32[8 * 2] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_tail_ocp_fp4<8, 64>(x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

    // Compile-only: exercise the half / fp32 input branches (not res-checked).
    dynamic_mx_quant_tail_ocp_fp4<8, 64, 32, __fp4_e2m1x2, __half>(
        xh16, reinterpret_cast<__fp4_e2m1x2*>(y_h16), scale_h16);
    dynamic_mx_quant_tail_ocp_fp4<8, 64, 32, __fp4_e2m1x2, float>(
        xf32, reinterpret_cast<__fp4_e2m1x2*>(y_f32), scale_f32);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
