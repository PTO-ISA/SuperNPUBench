#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/quant/cast_back.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[512] __attribute__((aligned(4096))) = {};
static float sf[32] __attribute__((aligned(4096))) = {};
static float out[512] __attribute__((aligned(4096))) = {};

int main() {
    cast_back_per_token<__bf16, 16, 16>(x, sf, out);
    return 0;
}
