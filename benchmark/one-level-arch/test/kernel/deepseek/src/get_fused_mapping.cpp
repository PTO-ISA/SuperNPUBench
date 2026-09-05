#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/get_fused_mapping.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t topk_idx[256] __attribute__((aligned(4096))) = {};
static std::int32_t pos_to_expert[256] __attribute__((aligned(4096))) = {};
static std::int32_t pos_to_token[256] __attribute__((aligned(4096))) = {};
static std::int32_t pos_to_token_topk[256] __attribute__((aligned(4096))) = {};
static std::int32_t token_topk_to_pos[256] __attribute__((aligned(4096))) = {};
static std::int32_t expert_start[64] __attribute__((aligned(4096))) = {};
static std::int32_t expert_end[64] __attribute__((aligned(4096))) = {};
static std::int32_t num_tokens_per_expert[64] __attribute__((aligned(4096))) = {};

int main() {
    get_fused_mapping<16, 8, 32>(topk_idx, pos_to_expert, pos_to_token, pos_to_token_topk, token_topk_to_pos, expert_start, expert_end, num_tokens_per_expert);
    return 0;
}
