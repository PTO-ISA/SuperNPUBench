#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/normalization/rms_norm_binary/rms_norm_binary.hpp"

#include <cstdint>

namespace supernpu::multi_thread {

// Rows are independent. Workspace is also divided by row so no cache level is
// shared between PEs.
template <typename DType, int Rows, int Columns, int TileRows,
          int TileColumns, int PowerOfTwoColumns,
          std::uint32_t PeCount = kDefaultPeCount>
void rms_norm_binary(DType *input, DType *output, float *workspace,
                     float eps = 1e-6f) {
    using Partition = ContiguousPartition<Rows, PeCount>;
    constexpr int kRowsPerPe = static_cast<int>(Partition::kItemsPerPe);
    constexpr std::size_t kWorkspaceValuesPerRow =
        rms_bin::kMaxLevels * rms_bin::kWsCols;
    const std::size_t row_offset = Partition::item_offset();

    ::rms_norm_binary<DType, kRowsPerPe, Columns, TileRows,
                      TileColumns, PowerOfTwoColumns>(
        input + row_offset * Columns, output + row_offset * Columns,
        workspace + row_offset * kWorkspaceValuesPerRow, eps);
}

}  // namespace supernpu::multi_thread
