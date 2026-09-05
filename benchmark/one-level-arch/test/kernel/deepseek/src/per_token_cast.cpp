#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/quant/per_token_cast.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[512] __attribute__((aligned(4096))) = {};
static float out_sf[32] __attribute__((aligned(4096))) = {};
static __bf16 out[512] __attribute__((aligned(4096))) = {};

int main() {
    per_token_cast<16, 16>(x, 16, out_sf, out, 448.0f, 1e-4f);
    return 0;
}
