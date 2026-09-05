#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/group_count_aux_fi.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t topk_idx[256] __attribute__((aligned(4096))) = {};
static float out[64] __attribute__((aligned(4096))) = {};

int main() {
    aux_fi<16, 8, 32>(topk_idx, out, 8);
    return 0;
}
