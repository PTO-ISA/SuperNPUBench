#include "tadd_args.hpp"
#include "basic_op/element_wise/tadd_multithread.hpp"
#include <common/pto_launch.hpp>
#include <device/kernel_entry.hpp>

namespace {
using TaddArguments = supernpu::device::KernelArguments<kTaddKernelSchema, 5>;

uint64_t tadd(const TaddArguments& args) {
    if (!args.is(0, supernpu::KernelArgumentType::Buffer) ||
        !args.is(1, supernpu::KernelArgumentType::Buffer) ||
        !args.is(2, supernpu::KernelArgumentType::Buffer) ||
        !args.is(3, supernpu::KernelArgumentType::Buffer) ||
        !args.is(4, supernpu::KernelArgumentType::U64))
        return static_cast<uint64_t>(supernpu::KernelArgumentStatus::TypeMismatch);
    const uint64_t block = pto::GetBlockIdx();
    const uint64_t count = pto::GetBlockNum();
    if (!count || block >= count) return 1;

    reinterpret_cast<uint64_t*>(args.buffer(3))[block] = (count << 32) | block;
    auto* a = reinterpret_cast<float*>(args.buffer(0));
    auto* b = reinterpret_cast<float*>(args.buffer(1));
    auto* out = reinterpret_cast<float*>(args.buffer(2));
    for (uint64_t tile = block; tile < args.u64(4); tile += count) {
        const uint64_t offset = tile * kTileElements;
        tadd_tile<kTileRows, kTileCols>(out + offset, a + offset, b + offset);
    }
    return 0;
}
}  // namespace

SUPERNPU_DEFINE_KERNEL(bench_tadd, kTaddKernelSchema, 5, tadd)
