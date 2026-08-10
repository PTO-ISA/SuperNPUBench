#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8_bigbs.hpp"
using namespace supernpu::tile_isa::mxquant;

// Non-tail cuBLAS-FP8, LARGE-BlockSize branch (方案A: split reduce axis). BlockSize
// >=96 has NO valid TileN for the plain single-load kernel (fp8 alignment lower
// bound 32 collides with the TileSize upper on the single TileN axis); this branch
// tiles the reduce axis into R_sub=32 sub-chunks so R_sub*TileN=1024<=2048 (current
// fp32-roundtrip budget) holds for any BlockSize.
// Axis=128 (=BlockSize, numKb=1), Post=64 (TileN=32, numN=2), R_sub=32 (numSub=4).
// x=[128,64] bf16, y=[128,64] fp8 bytes. scale compact planar uint8 E8M0
// [scaleRows, Post]: scaleRows = evenAlign(Axis/128) = 2 -> [2,64] bytes.
static __bf16 x[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_cublas_fp8_bigbs<128, 64, 128>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
