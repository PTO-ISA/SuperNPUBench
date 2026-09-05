#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/reduce_fused.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[4096] __attribute__((aligned(4096))) = {};
static float topk_weights[128] __attribute__((aligned(4096))) = {};
static std::int32_t token_topk_to_pos[128] __attribute__((aligned(4096))) = {};
static float out[2048] __attribute__((aligned(4096))) = {};

int main() {
    reduce_fused<__bf16, float, 16, 64, 4, 32, 64>(x, topk_weights, token_topk_to_pos, out);
    return 0;
}
