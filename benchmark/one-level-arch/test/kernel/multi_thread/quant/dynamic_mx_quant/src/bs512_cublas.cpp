#include <common/pto_tileop.hpp>
#include <cstdint>
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_cublas_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

static __bf16 x[8 * 512] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 512] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale: [M, scaleCols] with scaleCols = evenAlign(K/BlockSize).
// M=8, K=512, BlockSize=512 -> numKb=1 -> scaleCols=2 (col 1 is zero padding).
static uint8_t scale[8 * 2] __attribute__((aligned(4096))) = {};

int main() {
    dynamic_mx_quant_tail_cublas_fp8<8, 512, 512>(x, reinterpret_cast<__fp8_e4m3*>(y), scale);
    return 0;
}
