#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/reduction/cumsum_rowvec.hpp"

#include <cstdint>

namespace supernpu::multi_thread {

template <typename DType, int Rows, int Columns, int TileRows,
          int TileColumns, std::uint32_t PeCount = kDefaultPeCount>
void cumsum_rowvec(DType *input, DType *output) {
    using Partition = ContiguousPartition<Rows, PeCount>;
    constexpr int kRowsPerPe = static_cast<int>(Partition::kItemsPerPe);
    const std::size_t row_offset = Partition::item_offset();

    ::cumsum_row_rand<DType, kRowsPerPe, Columns,
                      TileRows, TileColumns>(
        input + row_offset * Columns, output + row_offset * Columns);
}

}  // namespace supernpu::multi_thread
