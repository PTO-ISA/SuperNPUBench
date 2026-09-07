#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/concat/concat_gather.hpp"

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

template <typename DType, std::size_t MaxDimensions = 8,
          int InputElements, int OutputElements, int TileElements,
          std::size_t DataDimensions, std::size_t ConcatDimension,
          std::uint32_t PeCount = kDefaultPeCount>
void concat_gather(DType *input, DType *output, std::size_t *input_shape,
                   std::size_t *output_shape) {
    using Partition = ContiguousPartition<OutputElements, PeCount>;
    constexpr int kElementsPerPe =
        static_cast<int>(Partition::kItemsPerPe);
    const std::size_t output_offset = Partition::item_offset();

    ::concat_gather<DType, MaxDimensions, InputElements, kElementsPerPe,
                    TileElements, DataDimensions, ConcatDimension>(
        input, output + output_offset, input_shape, output_shape,
        static_cast<std::uint32_t>(output_offset));
}

}  // namespace supernpu::multi_thread
