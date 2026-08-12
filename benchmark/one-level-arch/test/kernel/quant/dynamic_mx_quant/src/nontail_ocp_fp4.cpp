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

// Large-BlockSize (BS=128) throwaway buffers: no legal plain TileN at BS=128, so
// this call auto-routes to the 方案A split-reduce _bigbs path. Compile-only
// coverage (runtime blocked by toolchain<->emulator skew). Axis=128 (=1 reduce
// block), Post=64 -> y=[128,32] packed bytes, scale scaleRows=evenAlign(1)=2 -> [2,64].
static __bf16 x_bs128[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y_bs128[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_bs128[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_ocp_fp4<32, 64>(x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

    // Compile-only: instantiate the bigbs auto-route at BS=128 (not res-checked).
    dynamic_mx_quant_nontail_ocp_fp4<128, 64, 128>(
        x_bs128, reinterpret_cast<__fp4_e2m1x2*>(y_bs128), scale_bs128);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
