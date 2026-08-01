#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail.hpp"
using namespace supernpu::tile_isa::mxquant;

static __bf16 x[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
static uint16_t scale[8 * 32] __attribute__((aligned(4096))) = {};

int main() {
    dynamic_mx_quant_nontail<2, 32, 4, ScaleAlg::DYNAMIC_RANGE>(x, reinterpret_cast<__fp8_e4m3*>(y), scale);
    return 0;
}
