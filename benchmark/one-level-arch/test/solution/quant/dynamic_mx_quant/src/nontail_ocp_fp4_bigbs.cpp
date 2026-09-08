#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// Non-tail OCP-FP4, LARGE BlockSize (=128). Formerly routed to a 方案A split-reduce
// `_bigbs` kernel because a single [BlockSize, TileN] block exceeded the old 8KB tile
// ceiling; that kernel is now RETIRED. The unified public entry
// dynamic_mx_quant_nontail_ocp_fp4<Axis, Post, BlockSize> loads the whole [128,64]
// block in one tile (16KB, within the 256KB TilesizeCode ceiling) — TileN falls back
// to align=64 for this large BlockSize. Verified byte-exact == the old bigbs output.
// Axis=128 (=BlockSize, numKb=1), Post=64 (TileN=64=2 MX blocks). x=[128,64],
// y=[128,32] bytes. scale compact planar uint8 E8M0 [scaleRows, Post]:
// scaleRows = evenAlign(Axis/128) = 2 -> [2,64] bytes.
//
// SINGLE-PE: numKb=1 leaves no block-row parallelism, so kPeNum=1 (tid 0 does all;
// consistent with the former bigbs single-PE harness).
static __bf16 x[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_ocp_fp4<128, 64, 128, __fp4_e2m1x2, __bf16, /*kPeNum=*/1>(
        x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
