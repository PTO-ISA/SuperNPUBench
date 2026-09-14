#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// fp16 (__half) INPUT as the PRIMARY data path (not a compile-only side branch).
// Mirrors tail_cublas_fp8.cpp dims (M=8, K=32, BlockSize=32) but drives the
// InT=__half if-constexpr branch as the main call. Motivation: the cuBLAS scale
// path reduces the block amax in the INPUT dtype domain (TABS + TROWMAX on InT,
// dynamic_mx_quant_common.hpp:702/704); with fp16 input the emulator's TEPL
// dtype whitelist accepts FP16 for both TABS and TROWMAX, so this input type
// side-steps the BF16-TABS / BF16-TROWMAX emulator gaps.
static __half x[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale: [M, scaleCols], scaleCols = evenAlign(K/BlockSize).
// M=8, K=32, BlockSize=32 -> numKb=1 -> scaleCols=2 (col 1 is zero padding).
static uint8_t scale[8 * 2] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_tail_cublas_fp8<8, 32, 32, __fp8_e4m3, __half>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
