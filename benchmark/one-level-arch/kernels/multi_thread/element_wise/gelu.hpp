#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/element_wise/gelu.hpp"

#include <cstdint>

namespace supernpu::multi_thread {

// Split a flat GELU tensor into equal contiguous ranges. Each PE reuses the
// single-PE Tile kernel on its private input/output range.
template <typename DType, int Elements, int TileElements,
          std::uint32_t PeCount = kDefaultPeCount>
void gelu(DType *input, DType *output, bool approximate = false) {
    using Partition = ContiguousPartition<Elements, PeCount>;
    constexpr int kElementsPerPe =
        static_cast<int>(Partition::kItemsPerPe);
    static_assert(TileElements > 0);

    const std::size_t offset = Partition::item_offset();
    ::gelu<DType, kElementsPerPe, TileElements>(
        input + offset, output + offset, approximate);
}

}  // namespace supernpu::multi_thread
