#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/gather/gather.hpp"

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

// Partition the output/index rows. The lookup table is read-only and shared;
// every PE owns a contiguous range of indexes and output rows.
template <typename DType = float, typename IndexType = std::uint32_t,
          std::size_t TableRows, std::size_t OutputRows,
          std::size_t Columns, std::size_t TileRows, std::size_t TileColumns,
          std::uint32_t PeCount = kDefaultPeCount>
void gather(DType *table, IndexType *indexes, DType *output) {
    using Partition = ContiguousPartition<OutputRows, PeCount>;
    constexpr std::size_t kRowsPerPe = Partition::kItemsPerPe;
    const std::size_t row_offset = Partition::item_offset();

    ::gather<DType, IndexType, TableRows, kRowsPerPe, Columns,
             TileRows, TileColumns>(table, indexes + row_offset,
                                    output + row_offset * Columns);
}

}  // namespace supernpu::multi_thread
