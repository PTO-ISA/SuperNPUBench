#pragma once

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {
namespace detail {

// Static path helpers: tile register arrays require compile-time indices.
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

// Dynamic path helpers: physical tile columns stay compile-time, while the
// valid column count and the owned-segment range are runtime values.
// Twelve segments (three accumulators per PE) is the largest group that keeps
// the tile array in registers on the current toolchain.
constexpr int kSegmentsPerGroup = 12;
constexpr int kOwnedPerGroup = kSegmentsPerGroup / 4;

// Ordinary TSTORE does not emit B.DATR, so a partial tail store would fail the
// model's "PadValue=Zero" legality check. Emit the explicit padded form.
template <typename Gm, typename TileT>
inline void tstore_padded(Gm &dst, TileT &src, int valid_col) {
    asm volatile(
        "BSTART.TLSU TSTORE, %D[SrcType]\n"
        "B.DATR NORM, Zero\n"
        "B.DIM %[VCOL], 0, ->lb0\n"
        "B.DIM zero, %c[VROW], ->lb1\n"
        "B.DIM zero, %c[COL], ->lb2\n"
        "B.IOT %[s0], mask=1111, last\n"
        "B.IOR [%[d0],%[GmStride]], []\n"
        :
        : [d0] "r"(dst.data()), [s0] "Tr"(src.data()),
          [SrcType] "i"(type_traits<typename TileT::DType>::TypeCode),
          [VCOL] "r"(valid_col), [VROW] "i"(TileT::ValidRow),
          [COL] "i"(TileT::Cols),
          [GmStride] "r"(dst.GetStrideBytes(3))
        : "memory");
}

template <int I, typename TileT, int N>
inline void init_tiles(TileT (&tiles)[N], int valid_col) {
    tiles[I].ColMaskInternal = valid_col;
    if constexpr (I + 1 < N) {
        init_tiles<I + 1>(tiles, valid_col);
    }
}

template <int I, typename TileT, int N>
inline void zero_owned_range(TileT (&tiles)[N], int base, int segments,
                             int tid) {
    const int segment = base + tid + I * 4;
    if (segment < segments) {
        TEXPANDS(tiles[I], static_cast<typename TileT::DType>(0.0f));
    }
    if constexpr (I + 1 < N) {
        zero_owned_range<I + 1>(tiles, base, segments, tid);
    }
}

template <int I, typename TileT, int N>
inline void store_owned_range(TileT (&tiles)[N], int base, int segments,
                              int tid, int feature_tile, int features,
                              int tile_cols, typename TileT::DType *output) {
    const int segment = base + tid + I * 4;
    if (segment < segments) {
        using Gm = pto::global_tensor<typename TileT::DType,
                                      pto::RowMajor<-1, -1>>;
        Gm dst(output + static_cast<std::int64_t>(segment) * features +
                       static_cast<std::int64_t>(feature_tile) * tile_cols,
               segments, features);
        tstore_padded(dst, tiles[I],
                      static_cast<int>(tiles[I].GetValidCol()));
    }
    if constexpr (I + 1 < N) {
        store_owned_range<I + 1>(tiles, base, segments, tid, feature_tile,
                                 features, tile_cols, output);
    }
}

}  // namespace detail

// Static-shape entry: Rows, Features and Segments are compile-time constants.
// Segment s is owned by PE (s & 3) and accumulated in slot (s >> 2), so every
// output element has exactly one writer: no GM atomics or cross-PE barrier.
// IDs outside [0, Segments) are ignored; empty segments are written as zero.
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

// Dynamic-shape entry: rows, features and segments are runtime values.
//
// Shape contract (matching ops-nn UnsortedSegmentSum):
//   - data is [rows, features]; segment_ids is [rows]; num_segments is scalar.
//   - rows, features and segments may be any non-negative runtime values;
//     features does not need to be a multiple of the tile width (the last
//     feature tile uses a runtime valid column).
//   - data.shape must start with segment_ids.shape; for a 2D kernel the caller
//     flattens leading ids dimensions into rows and trailing data dimensions
//     into features.
//   - dimensions must fit int because the model's global_tensor constructor
//     takes runtime rows/columns as int.
//
// Segments are processed in groups of detail::kSegmentsPerGroup so the tile
// register array stays statically sized while `segments` remains unbounded.
template <typename DType, typename IdType = std::int32_t>
void unsorted_segment_sum(DType *data, const IdType *segment_ids,
                          DType *output, int rows, int features, int segments) {
    static_assert(std::is_same_v<DType, float> ||
                      std::is_same_v<DType, __half> ||
                      std::is_same_v<DType, __bf16> ||
                      std::is_same_v<DType, std::int32_t> ||
                      std::is_same_v<DType, std::uint32_t> ||
                      std::is_same_v<DType, std::int64_t> ||
                      std::is_same_v<DType, std::uint64_t>,
                  "dtype must match the ops-nn UnsortedSegmentSum contract");
    static_assert(std::is_same_v<IdType, std::int32_t> ||
                      std::is_same_v<IdType, std::int64_t>,
                  "segment_ids must be int32 or int64");

    if (features <= 0 || segments <= 0) {
        return;
    }

    constexpr int kTileCols = 256 / sizeof(DType);
    constexpr int kOwnedPerGroup = detail::kOwnedPerGroup;
    const int feature_tiles =
        static_cast<int>((static_cast<std::int64_t>(features) + kTileCols - 1) /
                         kTileCols);
    const int tid = static_cast<int>(get_thread_idx());

    using Gm = pto::global_tensor<DType, pto::RowMajor<-1, -1>>;
    using RowTile = pto::Tile<pto::Location::Vec, DType, 1, kTileCols,
                              pto::BLayout::RowMajor, 1, -1,
                              pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;

    for (int feature_tile = 0; feature_tile < feature_tiles; ++feature_tile) {
        const int remaining = features - feature_tile * kTileCols;
        const int valid_col = remaining < kTileCols ? remaining : kTileCols;

        for (int base = 0; base < segments; base += detail::kSegmentsPerGroup) {
            if (base + tid >= segments) {
                continue;
            }

            RowTile sums[kOwnedPerGroup];
            detail::init_tiles<0>(sums, valid_col);
            detail::zero_owned_range<0>(sums, base, segments, tid);

            for (int row = 0; row < rows; ++row) {
                const IdType raw = segment_ids[row];
                if (raw < static_cast<IdType>(0) ||
                    raw >= static_cast<IdType>(segments)) {
                    continue;
                }
                const int segment = static_cast<int>(raw);
                if ((segment & 3) != tid || segment < base ||
                    segment >= base + detail::kSegmentsPerGroup) {
                    continue;
                }

                // Keep the load inside the ownership branch so the timing
                // model does not hoist it and issue unnecessary GM traffic.
                asm volatile("" ::: "memory");

                const int slot = (segment - base) >> 2;
                RowTile loaded;
                loaded.ColMaskInternal = valid_col;
                Gm src(data + static_cast<std::int64_t>(row) * features +
                               static_cast<std::int64_t>(feature_tile) *
                                   kTileCols,
                       rows, features);
                TLOAD(loaded, src);
                detail::accumulate_at<0>(slot, sums, loaded);
            }

            detail::store_owned_range<0>(sums, base, segments, tid, feature_tile,
                                         features, kTileCols, output);
        }
    }
}

}  // namespace supernpu::multi_thread
