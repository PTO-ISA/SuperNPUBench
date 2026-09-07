#pragma once

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

inline constexpr std::uint32_t kDefaultPeCount = 4;

template <std::size_t TotalItems,
          std::uint32_t PeCount = kDefaultPeCount>
struct ContiguousPartition {
    static_assert(PeCount > 0, "PeCount must be positive");
    static_assert(TotalItems > 0, "the partition must not be empty");
    static_assert(TotalItems % PeCount == 0,
                  "the work dimension must be divisible by PeCount");

    static constexpr std::size_t kItemsPerPe = TotalItems / PeCount;

    static std::uint32_t pe_id() {
        const std::uint32_t tid = get_thread_idx();
        return tid;
    }

    static std::size_t item_offset() {
        return static_cast<std::size_t>(pe_id()) * kItemsPerPe;
    }
};

}  // namespace supernpu::multi_thread
