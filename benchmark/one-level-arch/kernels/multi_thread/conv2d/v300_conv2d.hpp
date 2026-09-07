#pragma once

#include "multi_thread/utils/spmd_partition.hpp"
#include "single_thread/conv2d/v300_conv2d.hpp"

#include <cstdint>

namespace supernpu::multi_thread::conv2d {

// Assign output-spatial (M) tile blocks round-robin across PEs. Input and
// weights are shared read-only; every output tile has exactly one owner.
template <typename DType,
          int InChannels, int InHeight, int InWidth, int OutChannels,
          int TileM, int TileN, int TileK,
          std::uint32_t PeCount = kDefaultPeCount>
void conv2d_1x1(float *output, DType *input, DType *weight) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::conv2d::conv2d_1x1<
        DType, InChannels, InHeight, InWidth, OutChannels,
        TileM, TileN, TileK, static_cast<int>(PeCount)>(
            output, input, weight, tid);
}

}  // namespace supernpu::multi_thread::conv2d
