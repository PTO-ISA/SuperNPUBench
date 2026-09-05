#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/quant/cast_back.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[1024] __attribute__((aligned(4096))) = {};
static float sf[64] __attribute__((aligned(4096))) = {};
static float out[1024] __attribute__((aligned(4096))) = {};

int main() {
    cast_back_per_channel<__bf16, 16, 32, 16, 32>(x, sf, out);
    return 0;
}
