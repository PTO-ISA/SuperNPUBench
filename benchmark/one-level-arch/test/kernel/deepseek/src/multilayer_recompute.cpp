#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/multilayer_recompute.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float initial_residual[512] __attribute__((aligned(4096))) = {};
static float comb_mix_0[512] __attribute__((aligned(4096))) = {};
static float comb_mix_1[512] __attribute__((aligned(4096))) = {};
static float layer_input_0[512] __attribute__((aligned(4096))) = {};
static float layer_input_1[512] __attribute__((aligned(4096))) = {};
static float out_residual[512] __attribute__((aligned(4096))) = {};
static float* comb_mix_ptrs[2] = {comb_mix_0, comb_mix_1};
static float* layer_input_ptrs[2] = {layer_input_0, layer_input_1};

int main() {
    multilayer_recompute<16, 16, 2>(initial_residual, comb_mix_ptrs, layer_input_ptrs, out_residual);
    return 0;
}
