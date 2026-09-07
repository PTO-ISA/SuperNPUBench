#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/transpose/transpose.hpp"

#include <cstdint>

namespace supernpu::multi_thread {

// Assign input row tiles round-robin. Transpose maps them to disjoint output
// column tiles. PE0 owns the single bottom-tail row tile when one exists.
template <typename DType, int Rows, int Columns, int TileRows = 16,
          int TileColumns = 16,
          std::uint32_t PeCount = kDefaultPeCount>
void transpose_2d(DType *input, DType *output) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::tile_transpose_2d<
        DType, Rows, Columns, TileRows, TileColumns,
        static_cast<int>(PeCount)>(input, output, tid, tid == 0);
}

}  // namespace supernpu::multi_thread
