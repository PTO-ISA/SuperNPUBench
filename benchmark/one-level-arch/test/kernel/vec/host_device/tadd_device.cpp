#include "tadd_args.hpp"
#include "basic_op/element_wise/tadd_multithread.hpp"
#include <common/pto_launch.hpp>

extern "C" uint64_t bench_tadd(uint64_t address) {
    const auto& args = *reinterpret_cast<const TaddArgs*>(address);
    const uint64_t block = pto::GetBlockIdx();
    const uint64_t count = pto::GetBlockNum();
    if (!count || block >= count) return 1;

    reinterpret_cast<uint64_t*>(args.geometry)[block] = (count << 32) | block;
    auto* a = reinterpret_cast<float*>(args.a);
    auto* b = reinterpret_cast<float*>(args.b);
    auto* out = reinterpret_cast<float*>(args.out);
    for (uint64_t tile = block; tile < args.tiles; tile += count) {
        const uint64_t offset = tile * kTileElements;
        tadd_tile<kTileRows, kTileCols>(out + offset, a + offset, b + offset);
    }
    return 0;
}
