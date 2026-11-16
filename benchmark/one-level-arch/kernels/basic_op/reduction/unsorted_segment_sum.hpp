#pragma once

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {
namespace detail {

// Tile register arrays need compile-time indices, so the per-owned-segment
// accumulator array is expanded through these helpers instead of a runtime loop.
template <int I, typename TileT, int N>
inline void expand_zero_at(TileT (&sums)[N]) {
    TEXPANDS(sums[I], static_cast<typename TileT::DType>(0.0f));
    if constexpr (I + 1 < N) {
        expand_zero_at<I + 1>(sums);
    }
}

template <int I, typename TileT, int N>
inline void accumulate_at(int slot, TileT (&sums)[N], TileT &value) {
    if (slot == I) {
        TADD(sums[I], sums[I], value);
    } else if constexpr (I + 1 < N) {
        accumulate_at<I + 1>(slot, sums, value);
    }
}

template <int I, typename TileT, int N, typename OutputIterator>
inline void store_at(int tid, int segments, int feature_tile,
                     TileT (&sums)[N], OutputIterator &output_it) {
    const int segment = tid + I * 4;
    if (segment < segments) {
        auto dst = output_it(segment, feature_tile);
        TSTORE(dst, sums[I]);
    }
    if constexpr (I + 1 < N) {
        store_at<I + 1>(tid, segments, feature_tile, sums, output_it);
    }
}

}  // namespace detail

// Deterministic four-PE unsorted segment sum.
//
// Segment s is owned by PE (s & 3) and accumulated in slot (s >> 2), so every
// output element has exactly one writer: repeated IDs need neither GM atomics
// nor a cross-PE barrier. IDs outside [0, Segments) are ignored, and empty
// segments are written as zero.
//
// Input and output share DType, matching the ops-nn UnsortedSegmentSum dtype
// contract (fp16/fp32/bf16/int32/uint32/int64/uint64, output.dtype == input).
template <typename DType, int Rows, int Features, int Segments>
void unsorted_segment_sum(DType *data, const std::int32_t *segment_ids,
                          DType *output) {
    static_assert(std::is_same_v<DType, float> ||
                      std::is_same_v<DType, __half> ||
                      std::is_same_v<DType, __bf16> ||
                      std::is_same_v<DType, std::int32_t> ||
                      std::is_same_v<DType, std::uint32_t> ||
                      std::is_same_v<DType, std::int64_t> ||
                      std::is_same_v<DType, std::uint64_t>,
                  "dtype must match the ops-nn UnsortedSegmentSum contract");
    static_assert(Rows > 0 && Segments > 0, "nonempty shapes required");
    static_assert(Features > 0 && Features % 32 == 0,
                  "feature count must be a positive multiple of 32");

    // A tile holds at most 256 B, so the column count follows the element width.
    constexpr int kMaxTileCols = 256 / sizeof(DType);
    constexpr int kTileCols = (Features % kMaxTileCols == 0)
                                  ? kMaxTileCols
                                  : ((Features % 64 == 0) ? 64 : 32);
    constexpr int kFeatureTiles = Features / kTileCols;
    constexpr int kOwnedSegments = (Segments + 3) / 4;

    using Input = pto::global_tensor<DType, pto::RowMajor<Rows, Features>>;
    using Output = pto::global_tensor<DType, pto::RowMajor<Segments, Features>>;
    using RowTile = pto::Tile<pto::Location::Vec, DType, 1, kTileCols,
                              pto::BLayout::RowMajor>;
    using InputIterator = global_iterator<Input, RowTile>;
    using OutputIterator = global_iterator<Output, RowTile>;

    InputIterator input_it(data);
    OutputIterator output_it(output);
    const int tid = static_cast<int>(get_thread_idx());

    for (int feature_tile = 0; feature_tile < kFeatureTiles; ++feature_tile) {
        RowTile sums[kOwnedSegments];
        detail::expand_zero_at<0>(sums);

        // Single pass over the rows; only the owning PE loads a row tile.
        for (int row = 0; row < Rows; ++row) {
            const int segment = segment_ids[row];
            if ((static_cast<unsigned>(segment) >=
                 static_cast<unsigned>(Segments)) ||
                ((segment & 3) != tid)) {
                continue;
            }

            // Keep the load inside the ownership branch so the timing model
            // does not hoist it and issue unnecessary GM traffic.
            asm volatile("" ::: "memory");

            const int slot = segment >> 2;
            RowTile loaded;
            auto src = input_it(row, feature_tile);
            TLOAD(loaded, src);
            detail::accumulate_at<0>(slot, sums, loaded);
        }

        detail::store_at<0>(tid, Segments, feature_tile, sums, output_it);
    }
}

}  // namespace supernpu::multi_thread
