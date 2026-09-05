#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/normalize_weight.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float topk_weights[256] __attribute__((aligned(4096))) = {};
static float denominator[32] __attribute__((aligned(4096))) = {};
static float normalized_weights[256] __attribute__((aligned(4096))) = {};

int main() {
    normalize_weight<16, 8>(topk_weights, denominator, normalized_weights);
    return 0;
}
