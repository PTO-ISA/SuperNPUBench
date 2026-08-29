#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/reduction/reducemax_rowvec.hpp"

#include <cstdint>

namespace supernpu::multi_thread {

template <typename DType, int Rows, int Columns, int TileRows,
          int TileColumns, std::uint32_t PeCount = kDefaultPeCount>
void reducemax_rowvec(DType *input, DType *output) {
    using Partition = ContiguousPartition<Rows, PeCount>;
    constexpr int kRowsPerPe = static_cast<int>(Partition::kItemsPerPe);
    const std::size_t row_offset = Partition::item_offset();

    ::reducemax_row_rand<DType, kRowsPerPe, Columns,
                         TileRows, TileColumns>(
        input + row_offset * Columns, output + row_offset);
}

}  // namespace supernpu::multi_thread
