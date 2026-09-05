#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/moe/mask_indices_by_tp.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t indices[256] __attribute__((aligned(4096))) = {};

int main() {
    mask_indices_by_tp<16, 8>(indices, 1, 1, 1, 0);
    return 0;
}
