#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/expand_to_mhc_bwd.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 o_grad[32768] __attribute__((aligned(4096))) = {};
static __bf16 x_grad[2048] __attribute__((aligned(4096))) = {};

int main() {
    expand_to_mhc_bwd<16, 64, 16, 64>(o_grad, x_grad);
    return 0;
}
