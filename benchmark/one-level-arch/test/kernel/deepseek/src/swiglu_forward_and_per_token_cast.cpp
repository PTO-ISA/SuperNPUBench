#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/quant/swiglu_fused_cast.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[1024] __attribute__((aligned(4096))) = {};
static __bf16 out[512] __attribute__((aligned(4096))) = {};
static float out_sf[32] __attribute__((aligned(4096))) = {};

int main() {
    swiglu_forward_and_per_token_cast<16, 16, 16>(x, out, out_sf, 448.0f, 1e-4f);
    return 0;
}
