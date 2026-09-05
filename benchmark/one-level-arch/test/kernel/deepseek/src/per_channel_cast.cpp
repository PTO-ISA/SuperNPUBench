#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/quant/per_token_cast.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[1024] __attribute__((aligned(4096))) = {};
static __bf16 out[1024] __attribute__((aligned(4096))) = {};
static float out_sf[64] __attribute__((aligned(4096))) = {};

int main() {
    per_channel_cast<16, 32, 32>(x, out, out_sf, 448.0f, 1e-4f);
    return 0;
}
