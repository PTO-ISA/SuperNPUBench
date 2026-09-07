#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/concat/concat_scatter.hpp"

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

template <typename DType, std::size_t MaxDimensions = 8,
          int InputElements, int OutputElements, int TileElements,
          std::size_t DataDimensions, std::size_t ConcatDimension,
          std::uint32_t PeCount = kDefaultPeCount>
void concat_scatter(DType *input, DType *output, std::size_t *input_shape,
                    std::size_t *output_shape) {
    using Partition = ContiguousPartition<InputElements, PeCount>;
    constexpr int kElementsPerPe =
        static_cast<int>(Partition::kItemsPerPe);
    const std::size_t input_offset = Partition::item_offset();

    ::concat_scatter<DType, MaxDimensions, kElementsPerPe, OutputElements,
                     TileElements, DataDimensions, ConcatDimension>(
        input + input_offset, output, input_shape, output_shape,
        static_cast<std::uint32_t>(input_offset));
}

}  // namespace supernpu::multi_thread
