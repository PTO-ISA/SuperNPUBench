#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/expand_to_fused.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static __bf16 x[2048] __attribute__((aligned(4096))) = {};
static std::int32_t token_topk_to_pos[128] __attribute__((aligned(4096))) = {};
static std::int32_t pos_to_expert[64] __attribute__((aligned(4096))) = {};
static __bf16 expanded_x[4096] __attribute__((aligned(4096))) = {};

int main() {
    expand_to_fused<__bf16, 16, 64, 4, 32, 64>(x, token_topk_to_pos, pos_to_expert, expanded_x);
    return 0;
}
