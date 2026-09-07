#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/broadcast/broadcast.hpp"

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

namespace detail {

// A broadcast that expands only the leading unit-dimension prefix produces
// complete, contiguous repetitions of the flattened input. When one
// repetition is assigned to each PE, use a direct Tile copy and avoid the
// generic MGATHER offset generator (and its backend raw-tile spill).
template <typename DType, std::size_t Elements, std::size_t TileElements>
void copy_full_input(DType *input, DType *output) {
    constexpr std::size_t kBlocks = Elements / TileElements;
    constexpr std::size_t kTail = Elements % TileElements;
    using Gm = global_tensor<DType, RowMajor<1, Elements>>;
    using TileT = Tile<Location::Vec, DType, 1, TileElements,
                       BLayout::RowMajor>;
    using Iterator = global_iterator<Gm, TileT>;
    Iterator input_iter(input);
    Iterator output_iter(output);

    for (std::size_t block = 0; block < kBlocks; ++block) {
        TileT tile;
        auto src = input_iter(0, block);
        auto dst = output_iter(0, block);
        TLOAD(tile, src);
        TSTORE(dst, tile);
    }
    if constexpr (kTail != 0) {
        using TailTile = Tile<Location::Vec, DType, 1, TileElements,
                              BLayout::RowMajor, 1, kTail>;
        using TailIterator = global_iterator<Gm, TailTile>;
        TailIterator input_tail_iter(input);
        TailIterator output_tail_iter(output);
        TailTile tile;
        auto src = input_tail_iter(0, kBlocks);
        auto dst = output_tail_iter(0, kBlocks);
        TLOAD(tile, src);
        TSTORE(dst, tile);
    }
}

}  // namespace detail

// Partition the flat output domain. output_base tells the single-PE kernel
// which global output coordinates belong to the PE-local contiguous range.
template <typename DType, std::size_t MaxDimensions = 8,
          std::size_t InputDimensions, std::size_t OutputDimensions,
          std::size_t InputElements, std::size_t OutputElements,
          std::size_t TileElements,
          std::uint32_t PeCount = kDefaultPeCount>
void broadcast(DType *input, DType *output, const std::size_t *input_shape,
               const std::size_t *output_shape) {
    using Partition = ContiguousPartition<OutputElements, PeCount>;
    constexpr std::size_t kElementsPerPe = Partition::kItemsPerPe;
    const std::size_t output_offset = Partition::item_offset();

    if constexpr (OutputDimensions >= InputDimensions &&
                  kElementsPerPe == InputElements) {
        constexpr std::size_t kAlignment =
            OutputDimensions - InputDimensions;
        std::size_t first_payload_dimension = InputDimensions;
        for (std::size_t dimension = 0; dimension < InputDimensions;
             ++dimension) {
            if (input_shape[dimension] != 1) {
                first_payload_dimension = dimension;
                break;
            }
        }

        bool repeats_full_input = true;
        for (std::size_t dimension = first_payload_dimension;
             dimension < InputDimensions; ++dimension) {
            if (input_shape[dimension] !=
                output_shape[kAlignment + dimension]) {
                repeats_full_input = false;
            }
        }
        if (repeats_full_input) {
            detail::copy_full_input<DType, InputElements, TileElements>(
                input, output + output_offset);
            return;
        }
    }

    ::broadcast<DType, MaxDimensions, InputDimensions, OutputDimensions,
                InputElements, kElementsPerPe, TileElements>(
        input, output + output_offset, input_shape, output_shape,
        output_offset);
}

}  // namespace supernpu::multi_thread
