#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

static __bf16 x[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[32 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale, transposed layout [numKb, Post], one byte per block.
// Axis=32, Post=32, BlockSize=32 -> numKb=1 -> [1, 32]. Allocate [2, 32] for the
// even-pad block-row headroom (AscendC ceil_even(numKb)).
static uint8_t scale[2 * 32] __attribute__((aligned(4096))) = {};

// Large-BlockSize (BS=128) throwaway buffers: the unified entry has NO legal plain
// TileN at BS=128, so this call auto-routes to the 方案A split-reduce _bigbs path.
// Compile-only coverage of that route (runtime blocked by toolchain<->emulator skew).
// Axis=128 (=1 reduce block), Post=32 -> scaleRows=evenAlign(1)=2 -> scale[2,32].
static __bf16 x_bs128[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y_bs128[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale_bs128[2 * 32] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_cublas_fp8<32, 32>(x, reinterpret_cast<__fp8_e4m3*>(y), scale);

    // Compile-only: instantiate the bigbs auto-route at BS=128 (not res-checked).
    dynamic_mx_quant_nontail_cublas_fp8<128, 32, 128>(
        x_bs128, reinterpret_cast<__fp8_e4m3*>(y_bs128), scale_bs128);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
