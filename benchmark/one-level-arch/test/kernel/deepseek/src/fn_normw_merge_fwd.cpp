#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/mhc/norm_fn.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float fn[1024] __attribute__((aligned(4096))) = {};
static float normw[64] __attribute__((aligned(4096))) = {};
static float out_fn[1024] __attribute__((aligned(4096))) = {};

int main() {
    fn_normw_merge_fwd<16, 32, 16, 32>(fn, normw, out_fn);
    return 0;
}
