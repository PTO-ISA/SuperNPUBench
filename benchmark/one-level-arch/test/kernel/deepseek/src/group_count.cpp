#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/group_count_aux_fi.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t group_idx[256] __attribute__((aligned(4096))) = {};
static std::int32_t out[64] __attribute__((aligned(4096))) = {};

int main() {
    group_count<16, 8, 32>(group_idx, out);
    return 0;
}
