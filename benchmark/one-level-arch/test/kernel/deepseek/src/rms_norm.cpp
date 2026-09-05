#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/norm_fn.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float x[256] __attribute__((aligned(4096))) = {};
static float out[256] __attribute__((aligned(4096))) = {};

int main() {
    rms_norm<16, 8>(x, out);
    return 0;
}
