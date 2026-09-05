#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// End-to-end RES_CHECK harness for the 方案A split-reduce _bigbs route. BlockSize=128
// (>= 96) has NO legal plain TileN, so the unified public entry
// dynamic_mx_quant_nontail_cublas_fp8<Axis, Post, BlockSize> auto-routes here via
// `if constexpr (TileN < align_lower)`. Kept SEPARATE from nontail_cublas_fp8.cpp
// (the plain BS=32 harness) so that already-verified plain path stays untouched.
//
// nontail: reduce along rows (Axis), Post is the free column axis.
//   Axis=128 (=1 reduce block, BlockSize=128), Post=32.
//   scaleRows = evenAlign(Axis/BlockSize) = evenAlign(1) = 2 -> scale[2, 32].
// golden: gen_dynamic_mx_quant_data.py --M 128 --K 32 --block-size 128
//         --algo CUBLAS --kernel nontail --dtype FP8 --scale-layout compact
static __bf16  x[128 * 32]     __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32]     __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 32]   __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    // Axis=128, Post=32, BlockSize=128 -> routes to _bigbs (方案A split-reduce).
    dynamic_mx_quant_nontail_cublas_fp8<128, 32, 128>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
