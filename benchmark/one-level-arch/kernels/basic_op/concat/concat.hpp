#pragma once

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace supernpu::multi_thread {

inline constexpr std::size_t kConcatMaxRank = 8;

struct ConcatShape {
    std::size_t rank;
    std::int64_t dims[kConcatMaxRank];
};

// All addresses are in the executing program's address space. Capacity is in
// elements, not bytes. Shapes describe contiguous row-major storage.
template <typename T>
struct ConcatTensor {
    const T *data;
    ConcatShape shape;
    std::size_t capacity;
};

enum class ConcatStatus : std::uint32_t {
    Success = 0, InvalidList, InvalidRank, InvalidAxis, InvalidShape,
    NullData, InsufficientCapacity, Overflow, OutputOverlap
};

namespace concat_list_detail {
inline bool ignored_empty(const ConcatShape &s) {
    return s.rank == 1 && s.dims[0] == 0;
}

inline bool multiply(std::size_t a, std::size_t b, std::size_t &out) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) return false;
    out = a * b;
    return true;
}

inline bool elements(const ConcatShape &s, std::size_t &out) {
    // An empty dimension makes the whole tensor empty, irrespective of order.
    for (std::size_t d = 0; d < s.rank; ++d) {
        if (s.dims[d] == 0) { out = 0; return true; }
    }
    out = 1;
    for (std::size_t d = 0; d < s.rank; ++d) {
        if (!multiply(out, static_cast<std::size_t>(s.dims[d]), out)) return false;
    }
    return true;
}

template <typename T, int TileElements>
inline void copy(const T *src, T *dst, std::size_t valid) {
    using Tile = pto::Tile<pto::Location::Vec, T, 1, TileElements,
                           pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
    // Only one row is accessed; the dynamic valid columns bound both transfers.
    using Global = pto::global_tensor<T, pto::RowMajor<1, TileElements>>;
    Global input(src);
    Global output(dst);
    Tile tile(valid);
    TLOAD(tile, input);
    TSTORE(output, tile);
}
}  // namespace concat_list_detail

// Each participating PE calls with the same immutable descriptors. The caller
// publishes inputs and joins PEs. inferred_shape, if provided, is PE-local.
// All validation precedes stores; errors leave the output and inferred_shape
// unchanged. Inputs may alias each other, but may not overlap the output.
template <typename T, int TileElements = 128, std::uint32_t PeCount = 4>
ConcatStatus concat(const ConcatTensor<T> *inputs, std::size_t count,
                    std::int64_t axis, T *output, std::size_t output_capacity,
                    ConcatShape *inferred_shape = nullptr) {
    using namespace concat_list_detail;
    static_assert(PeCount == 1 || PeCount == 4, "supported PE counts are 1 and 4");
    static_assert(TileElements > 0 && (TileElements & (TileElements - 1)) == 0,
                  "tile elements must be a power of two");
    static_assert(TileElements * sizeof(T) >= 128 && TileElements * sizeof(T) <= 2048,
                  "tile capacity must be 128..2048 bytes");
    if (!inputs) return ConcatStatus::InvalidList;
    std::size_t first = count;
    for (std::size_t i = 0; i < count; ++i) {
        const auto &s = inputs[i].shape;
        if (!s.rank || s.rank > kConcatMaxRank) return ConcatStatus::InvalidRank;
        for (std::size_t d = 0; d < s.rank; ++d)
            if (s.dims[d] < 0) return ConcatStatus::InvalidShape;
        if (!ignored_empty(s) && first == count) first = i;
    }
    // A5 aclnnCat filters one-dimensional [0] tensors before axis validation.
    if (first == count) {
        if (inferred_shape) *inferred_shape = ConcatShape{1, {0}};
        return ConcatStatus::Success;
    }
    ConcatShape shape = inputs[first].shape;
    const auto rank = static_cast<std::int64_t>(shape.rank);
    if (axis < -rank || axis >= rank) return ConcatStatus::InvalidAxis;
    if (axis < 0) axis += rank;
    shape.dims[axis] = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto &s = inputs[i].shape;
        if (ignored_empty(s)) continue;
        if (s.rank != shape.rank) return ConcatStatus::InvalidRank;
        for (std::size_t d = 0; d < s.rank; ++d)
            if (d != static_cast<std::size_t>(axis) && s.dims[d] != shape.dims[d])
                return ConcatStatus::InvalidShape;
        if (s.dims[axis] > std::numeric_limits<std::int64_t>::max() - shape.dims[axis])
            return ConcatStatus::Overflow;
        shape.dims[axis] += s.dims[axis];
    }
    std::size_t total;
    if (!elements(shape, total) || total > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(T))
        return ConcatStatus::Overflow;
    if (total > output_capacity) return ConcatStatus::InsufficientCapacity;
    if (total && !output) return ConcatStatus::NullData;
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const auto bytes = total * sizeof(T);
    if (out_addr > UINTPTR_MAX - bytes) return ConcatStatus::Overflow;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t n;
        if (!elements(inputs[i].shape, n) || n > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(T))
            return ConcatStatus::Overflow;
        if (n > inputs[i].capacity) return ConcatStatus::InsufficientCapacity;
        if (n && !inputs[i].data) return ConcatStatus::NullData;
        const auto address = reinterpret_cast<std::uintptr_t>(inputs[i].data);
        if (address > UINTPTR_MAX - n * sizeof(T)) return ConcatStatus::Overflow;
        if (n && total && address < out_addr + bytes && out_addr < address + n * sizeof(T))
            return ConcatStatus::OutputOverlap;
    }
    if (inferred_shape) *inferred_shape = shape;
    if (!total) return ConcatStatus::Success;
    std::size_t outer = 1, inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) outer *= shape.dims[d];
    for (std::size_t d = axis + 1; d < shape.rank; ++d) inner *= shape.dims[d];
    const std::size_t output_width = static_cast<std::size_t>(shape.dims[axis]) * inner;
    const std::uint32_t tid = get_thread_idx();
    if (tid >= PeCount) return ConcatStatus::Success;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (ignored_empty(inputs[i].shape)) continue;
        const std::size_t width = static_cast<std::size_t>(inputs[i].shape.dims[axis]) * inner;
        if (!width) continue;
        const std::size_t tiles = width / TileElements + (width % TileElements != 0);
        for (std::size_t job = tid; job < outer * tiles; job += PeCount) {
            const std::size_t row = job / tiles;
            const std::size_t col = (job % tiles) * TileElements;
            const std::size_t valid = width - col < TileElements ? width - col : TileElements;
            copy<T, TileElements>(inputs[i].data + row * width + col,
                                  output + row * output_width + offset + col, valid);
        }
        offset += width;
    }
    return ConcatStatus::Success;
}

}  // namespace supernpu::multi_thread
