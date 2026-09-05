#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/expand_to_mhc.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[2048] __attribute__((aligned(4096))) = {};
static __bf16 o[8192] __attribute__((aligned(4096))) = {};

int main() {
    expand_to_mhc_fwd<16, 64, 4>(x, o);
    return 0;
}
