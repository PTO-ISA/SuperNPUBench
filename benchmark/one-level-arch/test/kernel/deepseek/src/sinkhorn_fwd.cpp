#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/sinkhorn.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float comb_res_mix[1024] __attribute__((aligned(4096))) = {};
static float comb_res_mix_out[1024] __attribute__((aligned(4096))) = {};

int main() {
    sinkhorn_fwd<2, 16, 1>(comb_res_mix, comb_res_mix_out);
    return 0;
}
