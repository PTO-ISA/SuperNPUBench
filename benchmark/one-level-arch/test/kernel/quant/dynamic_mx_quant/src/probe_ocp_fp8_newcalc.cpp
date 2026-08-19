#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8_newcalc.hpp"
using namespace supernpu::tile_isa::mxquant;

// NEWCALC probe: half in -> e4m3 out, BlockSize=32, OCP; 倒数改用位补 (非 TRECIP)。
// 与 probe_ocp_fp8.cpp 同输入 x = 4.0 (half 0x4400), 期望 scale = 0x79 (2^-6),
// y = 0x78 (256 = 2^8, e4m3) —— 与原探针逐字节一致。
static uint16_t xbits[8 * 32] __attribute__((aligned(4096))) = {[0 ... 8 * 32 - 1] = 0x4400};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[8 * 1] __attribute__((aligned(4096))) = {};

int main() {
    probe_dynamic_mx_quant_tail_ocp_fp8_newcalc<8, 32, 32>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);
    return 0;
}
