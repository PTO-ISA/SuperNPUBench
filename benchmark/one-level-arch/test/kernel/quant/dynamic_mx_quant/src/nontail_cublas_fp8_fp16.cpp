#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// fp16 (__half) INPUT as the PRIMARY data path (not a compile-only side branch).
// Mirrors nontail_cublas_fp8.cpp dims (Axis=32, Post=32, BlockSize=32) but drives
// the InT=__half if-constexpr branch as the main call. The non-tail cuBLAS scale
// path reduces the block amax in the INPUT dtype domain (TABS + TCOLMAX on InT,
// dynamic_mx_quant_common.hpp:830/832); with fp16 input the emulator TEPL dtype
// whitelist accepts FP16 for TABS, side-stepping the BF16-TABS emulator gap.
static __half x[32 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[32 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale, transposed layout [numKb, Post], one byte per block.
// Axis=32, Post=32, BlockSize=32 -> numKb=1 -> [1,32]; allocate [2,32] for the
// even-pad block-row headroom (AscendC ceil_even(numKb)).
static uint8_t scale[2 * 32] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_cublas_fp8<32, 32, 32, __fp8_e4m3, __half>(
        x, reinterpret_cast<__fp8_e4m3*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
