#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// End-to-end RES_CHECK harness for LARGE BlockSize (=128) non-tail cuBLAS-FP8.
// Formerly routed to a 方案A split-reduce `_bigbs` kernel (now RETIRED); the unified
// public entry dynamic_mx_quant_nontail_cublas_fp8<Axis, Post, BlockSize> loads the
// whole [128,32] block in one tile (its 32b intermediates = 16KB, within the 256KB
// TilesizeCode ceiling) — TileN falls back to align=32 for this large BlockSize.
// Verified byte-exact == the old bigbs output. Kept SEPARATE from
// nontail_cublas_fp8.cpp (the BS=32 harness) so the small-BS case stays distinct.
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

    // Axis=128, Post=32, BlockSize=128 -> single-load plain (TileN=align=32).
    dynamic_mx_quant_nontail_cublas_fp8<128, 32, 128>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
