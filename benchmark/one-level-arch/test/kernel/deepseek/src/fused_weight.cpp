#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/engram/fused_weight.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 weight_hidden[256] __attribute__((aligned(4096))) = {};
static __bf16 weight_embed[256] __attribute__((aligned(4096))) = {};
static float weight_fused[256] __attribute__((aligned(4096))) = {};

int main() {
    fused_weight<2, 64, 64>(weight_hidden, weight_embed, weight_fused);
    return 0;
}
