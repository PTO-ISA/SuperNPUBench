#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail.hpp"
using namespace supernpu::tile_isa::mxquant;

static __bf16 x[8 * 256] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 256] __attribute__((aligned(4096))) = {};
static uint16_t scale[8 * 256] __attribute__((aligned(4096))) = {};

int main() {
    dynamic_mx_quant_tail<8, 256, ScaleAlg::OCP, 8, 64>(x, reinterpret_cast<__fp8_e4m3*>(y), scale);
    return 0;
}
