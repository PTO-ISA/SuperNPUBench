#include "fill_args.hpp"

#include <common/pto_launch.hpp>
#include <device/kernel_entry.hpp>

namespace {
using FillArguments = supernpu::device::KernelArguments<kFillKernelSchema, 4>;

uint64_t fill(const FillArguments& args) {
    if (!args.is(0, supernpu::KernelArgumentType::Buffer) ||
        !args.is(1, supernpu::KernelArgumentType::U32) ||
        !args.is(2, supernpu::KernelArgumentType::I32) ||
        !args.is(3, supernpu::KernelArgumentType::U64))
        return static_cast<uint64_t>(supernpu::KernelArgumentStatus::TypeMismatch);
    const uint64_t block = pto::GetBlockIdx();
    const uint64_t blocks = pto::GetBlockNum();
    if (!blocks || block >= blocks) return 1;

    auto* output = reinterpret_cast<int32_t*>(args.buffer(0));
    for (uint64_t index = block; index < args.u32(1); index += blocks)
        output[index] = args.i32(2) +
                        static_cast<int32_t>(index * args.u64(3));
    return 0;
}
}  // namespace

SUPERNPU_DEFINE_KERNEL(bench_fill, kFillKernelSchema, 4, fill)
