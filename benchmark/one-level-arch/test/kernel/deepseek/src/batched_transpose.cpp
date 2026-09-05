#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/transpose/batched_transpose.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float input[1024] __attribute__((aligned(4096))) = {};
static float output[1024] __attribute__((aligned(4096))) = {};

int main() {
    batched_transpose<float, 2, 16, 16>(input, output);
    return 0;
}
