#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/inplace_unique_group_indices.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t group_indices[256] __attribute__((aligned(4096))) = {};

int main() {
    inplace_unique_group_indices<16, 8, 8>(group_indices);
    return 0;
}
