#pragma once

#include <common/kernel_abi.hpp>

namespace supernpu::device {

template <std::uint32_t Schema, std::size_t ArgumentCount>
class KernelArguments {
public:
    static_assert(ArgumentCount <= UINT32_MAX / sizeof(KernelArgumentRecord),
                  "kernel argument list exceeds the descriptor ABI");
    static KernelArgumentStatus Validate(std::uint64_t address) {
        if (!address) return KernelArgumentStatus::NullDescriptor;
        if (address % alignof(KernelArgumentRecord) != 0)
            return KernelArgumentStatus::MisalignedDescriptor;
        const auto& header = *reinterpret_cast<const KernelArgumentHeader*>(address);
        if (header.magic != kKernelArgumentMagic)
            return KernelArgumentStatus::InvalidMagic;
        if (header.version != kKernelArgumentVersion)
            return KernelArgumentStatus::UnsupportedVersion;
        if (header.headerBytes != sizeof(KernelArgumentHeader))
            return KernelArgumentStatus::InvalidHeaderSize;
        if (header.payloadBytes != ArgumentCount * sizeof(KernelArgumentRecord))
            return KernelArgumentStatus::InvalidPayloadSize;
        if (header.schema != Schema) return KernelArgumentStatus::SchemaMismatch;
        return KernelArgumentStatus::Success;
    }

    explicit KernelArguments(std::uint64_t address)
        : records_(reinterpret_cast<const KernelArgumentRecord*>(
              address + sizeof(KernelArgumentHeader))) {}

    bool is(std::size_t index, KernelArgumentType type) const {
        if (index >= ArgumentCount || records_[index].reserved != 0 ||
            records_[index].type != static_cast<std::uint32_t>(type))
            return false;
        if (type == KernelArgumentType::U32 || type == KernelArgumentType::I32 ||
            type == KernelArgumentType::F32)
            return (records_[index].value >> 32) == 0;
        return true;
    }

    std::uint64_t buffer(std::size_t index) const { return records_[index].value; }
    std::uint32_t u32(std::size_t index) const {
        return static_cast<std::uint32_t>(records_[index].value);
    }
    std::int32_t i32(std::size_t index) const {
        return KernelArgumentI32(records_[index].value);
    }
    std::uint64_t u64(std::size_t index) const { return records_[index].value; }
    std::int64_t i64(std::size_t index) const {
        return KernelArgumentI64(records_[index].value);
    }
    float f32(std::size_t index) const {
        return KernelArgumentF32(records_[index].value);
    }
    double f64(std::size_t index) const {
        return KernelArgumentF64(records_[index].value);
    }

private:
    const KernelArgumentRecord* records_;
};

template <std::uint32_t Schema, std::size_t ArgumentCount>
inline std::uint64_t InvokeKernel(
    std::uint64_t address,
    std::uint64_t (*implementation)(const KernelArguments<Schema, ArgumentCount>&)) {
    const auto status = KernelArguments<Schema, ArgumentCount>::Validate(address);
    if (status != KernelArgumentStatus::Success)
        return static_cast<std::uint64_t>(status);
    return implementation(KernelArguments<Schema, ArgumentCount>(address));
}

}  // namespace supernpu::device

#define SUPERNPU_DEFINE_KERNEL(Name, Schema, ArgumentCount, Implementation)       \
    extern "C" std::uint64_t Name(std::uint64_t descriptor) {                  \
        return ::supernpu::device::InvokeKernel<Schema, ArgumentCount>(           \
            descriptor, Implementation);                                         \
    }
