#pragma once

#include "basic_op/reduction/detail/reduce_colvec.hpp"
#include "basic_op/utils/spmd_partition.hpp"

namespace supernpu::multi_thread {

template <typename DType, int Rows, int Columns, int TileRows,
          int TileColumns, std::uint32_t PeCount = kDefaultPeCount>
void reducemax_colvec(DType *input, DType *output) {
    reduce_colvec_pto<true, DType, Rows, Columns, TileRows, TileColumns,
                      PeCount>(input, output, get_thread_idx(), PeCount);
}

template <typename DType, int Batches, int Rows, int Columns,
          int TileRows, int TileColumns,
          std::uint32_t PeCount = kDefaultPeCount>
void reducemax_3dcol(DType *input, DType *output) {
    const std::uint32_t tid = get_thread_idx();
    for (int batch = static_cast<int>(tid); batch < Batches;
         batch += static_cast<int>(PeCount)) {
        reduce_colvec_pto<true, DType, Rows, Columns, TileRows, TileColumns,
                          1>(input + batch * Rows * Columns,
                             output + batch * Columns, 0, 1);
    }
}

}  // namespace supernpu::multi_thread
