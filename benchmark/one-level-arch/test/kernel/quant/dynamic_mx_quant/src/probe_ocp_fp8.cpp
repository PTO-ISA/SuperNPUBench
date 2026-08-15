#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// Minimal probe: half in -> e4m3 out, BlockSize=32, OCP new formula (Cast e8m0).
// M=8, N=32 -> single tile (8*32*2=512B), single block (numKb=1, no M_tail).
static __half  x[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale: [M, numKb] = [8, 1].
static uint8_t scale[8 * 1] __attribute__((aligned(4096))) = {};

int main() {
    probe_dynamic_mx_quant_tail_ocp_fp8<8, 32, 32>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);
    return 0;
}
