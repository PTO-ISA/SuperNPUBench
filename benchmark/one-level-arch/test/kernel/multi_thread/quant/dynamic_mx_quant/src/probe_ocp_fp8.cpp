#include <common/pto_tileop.hpp>
#include <cstdint>
#include "multi_thread/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp"
using namespace supernpu::tile_isa::mxquant;

// Minimal probe: half in -> e4m3 out, BlockSize=32, OCP new formula (Cast e8m0).
// M=8, N=32 -> single tile (8*32*2=512B), single block (numKb=1, no M_tail).
// 固定值精度探针：x = 4.0 (half 0x4400) 均匀填充，静态初始化避免运行期标量循环噪声。
// 期望 scale = 0x79 (2^-6), y = 0x78 (256 = 2^8, e4m3)。
static uint16_t xbits[8 * 32] __attribute__((aligned(4096))) = {[0 ... 8 * 32 - 1] = 0x4400};
static __half  *x = reinterpret_cast<__half *>(xbits);
static uint8_t y[8 * 32] __attribute__((aligned(4096))) = {};
// Compact uint8 E8M0 scale: [M, numKb] = [8, 1].
static uint8_t scale[8 * 1] __attribute__((aligned(4096))) = {};

int main() {
    probe_dynamic_mx_quant_tail_ocp_fp8<8, 32, 32>(
        x, reinterpret_cast<__fp8_e4m3 *>(y), scale);
    return 0;
}
