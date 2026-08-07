#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// FP4 output packed 2/byte along Post -> y holds Post/2 bytes per row.
// Post=64 (TileN=64 = 2 MX blocks) so the fp4 output tile [32,32] is a plain
// RowMajor NoneBox that satisfies the 32B column alignment; see RECORD problem 3.
// Axis=32, Post=64: x=[32,64], y=[32,32] bytes. scale is compact planar uint8
// E8M0 [scaleRows, Post]: scaleRows = evenAlign(Axis/32) = 2 -> [2,64] bytes.
static __bf16 x[32 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_ocp_fp4<32, 64>(x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
