#pragma once

#include "basic_op/utils/spmd_partition.hpp"
#include "basic_op/concat/detail/concat_scatter.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {

template <typename DType, std::size_t MaxDimensions = 8,
          int InputElements, int OutputElements, int TileElements,
          std::size_t DataDimensions, std::size_t ConcatDimension,
          std::uint32_t PeCount = kDefaultPeCount>
void concat_scatter(DType *input, DType *output, std::size_t *input_shape,
                    std::size_t *output_shape) {
    constexpr bool ElementProfile = DataDimensions == 2 && ConcatDimension == 1 &&
        (((std::is_same_v<DType, int32_t> || std::is_same_v<DType, __half>) &&
          TileElements == 512) ||
         (std::is_same_v<DType, float> && TileElements == 128));
    if constexpr (ElementProfile) {
        static_assert(PeCount > 0 && InputElements > 0 && OutputElements > 0,
                      "concat scatter requires a non-empty allocation and PE set");
        constexpr std::uint32_t kProcessed =
            InputElements < OutputElements ? InputElements : OutputElements;
        static_assert(kProcessed % TileElements == 0,
                      "concat scatter partitions complete original input Tiles");
        constexpr std::uint32_t kTiles = kProcessed / TileElements;
        const std::uint32_t pe = get_thread_idx();
        if (pe >= PeCount)
            __builtin_trap();
        // 按完整 Tile 平衡划分处理范围。FP16 的未使用输入尾部不属于工作量；
        // 最后一个 PE 可以少一个 Tile，仍保持各自的完整输入快照。
        const std::uint32_t first =
            (static_cast<std::uint64_t>(kTiles) * pe) / PeCount;
        const std::uint32_t last =
            (static_cast<std::uint64_t>(kTiles) * (pe + 1)) / PeCount;
        const std::uint32_t input_offset = first * TileElements;
        const std::uint32_t processed = (last - first) * TileElements;

        ::concat_scatter<DType, MaxDimensions, InputElements, OutputElements,
                         TileElements, DataDimensions, ConcatDimension>(
            input + input_offset, output, input_shape, output_shape,
            input_offset, processed);
    } else {
        using Partition = ContiguousPartition<InputElements, PeCount>;
        constexpr int kElementsPerPe = static_cast<int>(Partition::kItemsPerPe);
        const std::size_t input_offset = Partition::item_offset();
        ::concat_scatter<DType, MaxDimensions, kElementsPerPe, OutputElements,
                         TileElements, DataDimensions, ConcatDimension>(
            input + input_offset, output, input_shape, output_shape,
            static_cast<std::uint32_t>(input_offset));
    }
}

}  // namespace supernpu::multi_thread
