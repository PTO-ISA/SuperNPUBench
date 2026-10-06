#pragma once

#include "basic_op/utils/spmd_partition.hpp"

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

// Broadcast [Rows, 1] to [Rows, Copies] without MGATHER.  A tile contains one
// input column, and the ISA-visible broadcast is 1 TLOAD followed by Copies
// strided TSTOREs.  Blocks are assigned round-robin to the four PEs so this
// also supports row counts which are not divisible by the PE count.
template <typename DType, std::size_t Rows, std::size_t Copies,
          std::size_t TileRows,
          std::uint32_t PeCount = kDefaultPeCount>
void broadcast_vec_2d(DType *input, DType *output) {
    static_assert(Rows > 0 && Copies > 0 && TileRows > 0,
                  "broadcast dimensions must be positive");
    constexpr std::size_t kFullBlocks = Rows / TileRows;
    constexpr std::size_t kTailRows = Rows % TileRows;

    using InputGm = global_tensor<DType, RowMajor<Rows, 1>>;
    using OutputGm = global_tensor<DType, RowMajor<Rows, Copies>>;
    using DataTile = Tile<Location::Vec, DType, TileRows, 1,
                          BLayout::RowMajor>;
    using InputIterator = global_iterator<InputGm, DataTile>;
    using OutputIterator = global_iterator<OutputGm, DataTile>;

    InputIterator input_iter(input);
    OutputIterator output_iter(output);
    const std::uint32_t tid = get_thread_idx();

    for (std::size_t block = tid; block < kFullBlocks; block += PeCount) {
        DataTile data;
        auto src = input_iter(block, 0);
        TLOAD(data, src);
        #pragma clang loop unroll(full)
        for (std::size_t copy = 0; copy < Copies; ++copy) {
            auto dst = output_iter(block, copy);
            TSTORE(dst, data);
        }
    }

    if constexpr (kTailRows != 0) {
        if (tid == (kFullBlocks % PeCount)) {
            using TailTile = Tile<Location::Vec, DType, TileRows, 1,
                                  BLayout::RowMajor, kTailRows, 1>;
            using TailInputIterator = global_iterator<InputGm, TailTile>;
            using TailOutputIterator = global_iterator<OutputGm, TailTile>;
            TailInputIterator tail_input(input);
            TailOutputIterator tail_output(output);
            TailTile data;
            auto src = tail_input(kFullBlocks, 0);
            TLOAD(data, src);
            #pragma clang loop unroll(full)
            for (std::size_t copy = 0; copy < Copies; ++copy) {
                auto dst = tail_output(kFullBlocks, copy);
                TSTORE(dst, data);
            }
        }
    }
}

// Broadcast [Batches, 1, Inner] to [Batches, Copies, Inner] without MGATHER.
// One vector tile holds TileBatches input rows.  Repeated strided TSTOREs copy
// that tile into each output copy while preserving the row-major [B,C,K]
// layout.  Each PE owns a disjoint contiguous batch range.
template <typename DType, std::size_t Batches, std::size_t Copies,
          std::size_t Inner, std::size_t TileBatches,
          std::uint32_t PeCount = kDefaultPeCount>
void broadcast_vec_3d(DType *input, DType *output) {
    static_assert(Batches % PeCount == 0,
                  "Batches must be divisible by the PE count");
    constexpr std::size_t kBatchesPerPe = Batches / PeCount;
    static_assert(kBatchesPerPe % TileBatches == 0,
                  "each PE batch range must contain complete tiles");
    constexpr std::size_t kBlocksPerPe = kBatchesPerPe / TileBatches;

    using InputGm = global_tensor<
        DType, RowMajor<kBatchesPerPe, Inner>>;
    using OutputGm = global_tensor<
        DType, RowMajor<kBatchesPerPe, Copies * Inner>>;
    using DataTile = Tile<Location::Vec, DType, TileBatches, Inner,
                          BLayout::RowMajor>;
    using InputIterator = global_iterator<InputGm, DataTile>;
    using OutputIterator = global_iterator<OutputGm, DataTile>;

    const std::size_t batch_offset =
        static_cast<std::size_t>(get_thread_idx()) * kBatchesPerPe;
    InputIterator input_iter(input + batch_offset * Inner);
    OutputIterator output_iter(
        output + batch_offset * Copies * Inner);

    DataTile data;
    for (std::size_t block = 0; block < kBlocksPerPe; ++block) {
        auto src = input_iter(block, 0);
        TLOAD(data, src);

        #pragma clang loop unroll(full)
        for (std::size_t copy = 0; copy < Copies; ++copy) {
            auto dst = output_iter(block, copy);
            TSTORE(dst, data);
        }
    }
}

}  // namespace supernpu::multi_thread
