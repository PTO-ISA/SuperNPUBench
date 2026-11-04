#pragma once

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {
namespace concat_detail {

template <typename T, int TileElements, int ValidElements>
inline void copy_tile(T *input, T *output) {
    using Tile = pto::Tile<pto::Location::Vec, T, 1, TileElements,
                           pto::BLayout::RowMajor, 1, ValidElements>;
    using Global = pto::global_tensor<T, pto::RowMajor<1, ValidElements>>;
    Global src(input);
    Global dst(output);
    Tile data;
    TLOAD(data, src);
    TSTORE(dst, data);
}

template <typename T, std::size_t Outer, std::size_t Width,
          std::size_t OutputWidth, std::size_t OutputOffset,
          int TileElements, std::uint32_t PeCount>
inline void copy_input(T *input, T *output, std::uint32_t tid) {
    constexpr std::size_t full_tiles = Width / TileElements;
    constexpr int tail = Width % TileElements;
    // Distribute (row, tile) jobs, including when Outer is smaller than PeCount.
    // No job crosses a row boundary or the split between the two inputs.
    if constexpr (full_tiles != 0) {
        for (std::size_t job = tid; job < Outer * full_tiles; job += PeCount) {
            const std::size_t row = job / full_tiles;
            const std::size_t col = (job % full_tiles) * TileElements;
            copy_tile<T, TileElements, TileElements>(
                input + row * Width + col,
                output + row * OutputWidth + OutputOffset + col);
        }
    }
    if constexpr (tail != 0) {
        for (std::size_t row = tid; row < Outer; row += PeCount) {
            copy_tile<T, TileElements, tail>(
                input + row * Width + full_tiles * TileElements,
                output + row * OutputWidth + OutputOffset + full_tiles * TileElements);
        }
    }
}

}  // namespace concat_detail

// Concatenate two contiguous row-major tensors along a compile-time shape axis.
// Outer = product(shape[:axis]), Inner = product(shape[axis+1:]); the inputs
// have shapes [Outer, LeftAxis, Inner] and [Outer, RightAxis, Inner].
// Output must not overlap either input. All participating PEs call this function;
// the caller publishes input data and joins PEs before consuming output.
template <typename T, std::size_t Outer, std::size_t LeftAxis,
          std::size_t RightAxis, std::size_t Inner = 1,
          int TileElements = 128, std::uint32_t PeCount = 4>
void concat_two(T *left, T *right, T *output) {
    static_assert(Outer > 0 && Inner > 0, "Outer and Inner must be positive");
    static_assert(TileElements > 0 && (TileElements & (TileElements - 1)) == 0,
                  "TileElements must be a positive power of two");
    static_assert(TileElements * sizeof(T) <= 2048, "tile exceeds 2048 bytes");
    static_assert(PeCount == 1 || PeCount == 4, "supported PE counts are 1 and 4");
    constexpr std::size_t left_width = LeftAxis * Inner;
    constexpr std::size_t right_width = RightAxis * Inner;
    constexpr std::size_t output_width = left_width + right_width;
    const std::uint32_t tid = get_thread_idx();
    if (tid >= PeCount) {
        return;
    }
    concat_detail::copy_input<T, Outer, left_width, output_width, 0,
                              TileElements, PeCount>(left, output, tid);
    concat_detail::copy_input<T, Outer, right_width, output_width, left_width,
                              TileElements, PeCount>(right, output, tid);
}

}  // namespace supernpu::multi_thread
