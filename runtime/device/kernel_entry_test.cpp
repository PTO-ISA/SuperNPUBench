#include <device/kernel_entry.hpp>

#include <cassert>

namespace {
constexpr std::uint32_t kTestSchema = 0x54455354u;
using Arguments = supernpu::device::KernelArguments<kTestSchema, 2>;

std::uint64_t body(const Arguments& args) {
    if (!args.is(0, supernpu::KernelArgumentType::Buffer) ||
        !args.is(1, supernpu::KernelArgumentType::I32))
        return static_cast<std::uint64_t>(
            supernpu::KernelArgumentStatus::TypeMismatch);
    return args.buffer(0) + static_cast<std::uint64_t>(args.i32(1));
}

struct Descriptor {
    supernpu::KernelArgumentHeader header;
    supernpu::KernelArgumentRecord arguments[2];
};

Descriptor valid() {
    return {{supernpu::kKernelArgumentMagic, supernpu::kKernelArgumentVersion,
             sizeof(supernpu::KernelArgumentHeader),
             2 * sizeof(supernpu::KernelArgumentRecord), kTestSchema},
            {supernpu::MakeKernelArgument(supernpu::KernelArgumentType::Buffer, 40),
             supernpu::MakeKernelArgument(supernpu::KernelArgumentType::I32, 2)}};
}
}  // namespace

int main() {
    auto descriptor = valid();
    const auto address = reinterpret_cast<std::uint64_t>(&descriptor);
    assert((supernpu::device::InvokeKernel<kTestSchema, 2>(address, body) == 42));

    descriptor.header.magic = 0;
    assert((supernpu::device::InvokeKernel<kTestSchema, 2>(address, body) ==
            static_cast<std::uint64_t>(supernpu::KernelArgumentStatus::InvalidMagic)));
    descriptor = valid();
    descriptor.header.payloadBytes = 0;
    assert((supernpu::device::InvokeKernel<kTestSchema, 2>(address, body) ==
            static_cast<std::uint64_t>(
                supernpu::KernelArgumentStatus::InvalidPayloadSize)));
    descriptor = valid();
    descriptor.header.schema = 0;
    assert((supernpu::device::InvokeKernel<kTestSchema, 2>(address, body) ==
            static_cast<std::uint64_t>(supernpu::KernelArgumentStatus::SchemaMismatch)));
    descriptor = valid();
    descriptor.arguments[1].type =
        static_cast<std::uint32_t>(supernpu::KernelArgumentType::U32);
    assert((supernpu::device::InvokeKernel<kTestSchema, 2>(address, body) ==
            static_cast<std::uint64_t>(supernpu::KernelArgumentStatus::TypeMismatch)));
}
