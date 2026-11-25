#pragma once
#include "basic_op/transpose/transpose_tload.hpp"
#include "basic_op/utils/spmd_partition.hpp"
#include <array>
#include <cstdint>

namespace supernpu::multi_thread {
namespace transpose_plane_detail {
// A plane is a whole output suffix, or an exact subdivision of the last axis.
// Therefore input_offset(plane_base + j) = input_offset(plane_base) + input_offset(j).
template<auto Shape, auto Perm, unsigned MaxPlaneElements>
struct Plan {
    static constexpr unsigned rank = Shape.size();
    static constexpr auto shape = Shape;
    static constexpr auto perm = Perm;
    static_assert(rank >= 1 && rank <= 5 && Perm.size() == rank);
    static_assert(MaxPlaneElements >= 1 && MaxPlaneElements <= 4096);
    static_assert([] {
        std::uint64_t n = 1;
        for (unsigned i = 0; i < rank; ++i) {
            if (Shape[i] <= 0 || Perm[i] < 0 || Perm[i] >= rank) return false;
            for (unsigned j = 0; j < i; ++j)
                if (Perm[i] == Perm[j]) return false;
            n *= Shape[i];
            if (n > 0x3fffffffULL) return false;
        }
        return true;
    }(), "Invalid shape/permutation or tensor exceeds signed 32-bit byte offsets");
    static constexpr auto out_shape = [] {
        std::array<unsigned, rank> a{};
        for (unsigned i = 0; i < rank; ++i) a[i] = Shape[Perm[i]];
        return a;
    }();
    static constexpr auto in_stride = [] {
        std::array<unsigned, rank> a{};
        unsigned stride = 1;
        for (int i = rank - 1; i >= 0; --i) { a[i] = stride; stride *= Shape[i]; }
        return a;
    }();
    static constexpr auto out_stride = [] {
        std::array<unsigned, rank> a{};
        unsigned stride = 1;
        for (int i = rank - 1; i >= 0; --i) { a[i] = stride; stride *= out_shape[i]; }
        return a;
    }();
    static constexpr unsigned count = [] {
        unsigned n = 1; for (auto d : Shape) n *= d; return n;
    }();
    static constexpr unsigned plane_elements = [] {
        unsigned n = out_shape[rank - 1];
        if (n > MaxPlaneElements) {
            // Exact subdivisions preserve index reuse even for very long axes.
            for (unsigned d = MaxPlaneElements; d > 0; --d)
                if (n % d == 0) return d;
        }
        if constexpr (rank > 1)
            if (out_shape[rank - 2] <= MaxPlaneElements / n)
                n *= out_shape[rank - 2];
        return n;
    }();
    static constexpr unsigned input_offset(unsigned flat) {
        unsigned offset = 0;
        for (int d = rank - 1; d >= 0; --d) {
            offset += (flat % out_shape[d]) * in_stride[Perm[d]];
            flat /= out_shape[d];
        }
        return offset;
    }
};

// Locality-aware rectangular plan.  The output axis that was contiguous in
// the input becomes the M32 row coordinate; the output last axis becomes the
// column coordinate.  All remaining axes are outer loops.  This preserves the
// compile-time generic interface while giving MGATHER the same coordinates as
// a hand-written rectangular kernel.
template<class P>
struct RectPlan {
    static constexpr unsigned row_axis = [] {
        unsigned best = P::rank;
        unsigned best_stride = ~0u;
        for (unsigned d = 0; d + 1 < P::rank; ++d) {
            const unsigned stride = P::in_stride[P::perm[d]];
            if (stride < best_stride) { best = d; best_stride = stride; }
        }
        return best;
    }();
    static constexpr unsigned col_axis = P::rank - 1;
    static constexpr unsigned candidate_rows =
        row_axis < P::rank ? P::out_shape[row_axis] : 0;
    static constexpr unsigned candidate_cols = P::out_shape[col_axis];
    static constexpr unsigned candidate_src_row_stride =
        row_axis < P::rank ? P::in_stride[P::perm[row_axis]] : 0;
    static constexpr bool available = P::rank >= 2 && row_axis != col_axis &&
        candidate_src_row_stride == 1 && candidate_rows >= 16;
    static constexpr unsigned rows = available ? candidate_rows : 0;
    static constexpr unsigned cols = available ? candidate_cols : 0;
    static constexpr unsigned outer_count = available ? P::count / (rows * cols) : 0;
    static constexpr unsigned src_row_stride = available ? P::in_stride[P::perm[row_axis]] : 0;
    static constexpr unsigned src_col_stride = available ? P::in_stride[P::perm[col_axis]] : 0;
    static constexpr unsigned dst_row_stride = available ? P::out_stride[row_axis] : 0;

    __attribute__((always_inline)) static void outer_bases(
        unsigned flat, unsigned& input_base, unsigned& output_base) {
        input_base = 0;
        output_base = 0;
        for (int d = int(P::rank) - 1; d >= 0; --d) {
            if (unsigned(d) == row_axis || unsigned(d) == col_axis) continue;
            const unsigned coord = flat % P::out_shape[d];
            flat /= P::out_shape[d];
            input_base += coord * P::in_stride[P::perm[d]];
            output_base += coord * P::out_stride[d];
        }
    }
};

template<class P, unsigned Rows, unsigned Cols>
struct RectTransfer {
    using RP = RectPlan<P>;
    static constexpr unsigned pitch = (Cols + 1) / 2 * 2;
    static_assert(Rows > 0 && Rows <= 32 && Cols > 0 && Cols <= 32);
    // CUBE TSTORE must match the producer's valid rectangle exactly.
    // Keep the known-working full-tile gather path; short tiles use TLOAD
    // with the same Rows/Cols at production and store (no implicit crop).
    static constexpr bool useGather = Rows == 32 && Cols == 32;
    using Index = pto::Tile<pto::Location::Left, std::uint32_t, 32, Cols,
                            pto::BLayout::CubeM32, Rows, Cols>;
    using Data = pto::Tile<pto::Location::Vec, std::uint16_t, 32, pitch,
                           pto::BLayout::CubeM32, Rows, Cols>;
    template<unsigned RowStride> using GM = pto::GlobalTensor<std::uint16_t,
        pto::Shape<1,1,1,Rows,Cols>,
        pto::Stride<1,1,Rows*RowStride,RowStride,1>, pto::Layout::ND>;

    Index indices;

    __attribute__((always_inline)) RectTransfer() {
        if constexpr (useGather) {
            Index row, col, row_bytes, col_bytes;
            TCI_ROW(row, std::uint32_t(0));
            TCI_COL(col, std::uint32_t(0));
            TMULS(row_bytes, row, std::uint32_t(RP::src_row_stride * 2));
            TMULS(col_bytes, col, std::uint32_t(RP::src_col_stride * 2));
            TADD(indices, row_bytes, col_bytes);
        }
    }

    __attribute__((always_inline)) static void task_bases(
        unsigned task, unsigned row_base, unsigned& input_base,
        unsigned& output_base) {
        constexpr unsigned col_tiles = RP::cols / 32;
        const unsigned outer = task / col_tiles;
        const unsigned col_base = (task % col_tiles) * 32;
        RP::outer_bases(outer, input_base, output_base);
        input_base += row_base * RP::src_row_stride + col_base * RP::src_col_stride;
        output_base += row_base * RP::dst_row_stride + col_base;
    }

    __attribute__((always_inline)) static void tail_bases(
        unsigned outer, unsigned row_base, unsigned& input_base,
        unsigned& output_base) {
        RP::outer_bases(outer, input_base, output_base);
        constexpr unsigned col_base = (RP::cols / 32) * 32;
        input_base += row_base * RP::src_row_stride + col_base * RP::src_col_stride;
        output_base += row_base * RP::dst_row_stride + col_base;
    }

    __attribute__((always_inline)) void copy1(std::uint16_t* input,
                                               std::uint16_t* output,
                                               unsigned input_base,
                                               unsigned output_base) {
        if constexpr (!useGather) {
            for (unsigned c = 0; c < Cols; ++c)
                transpose_detail::copy<Rows, 1, RP::src_row_stride, RP::dst_row_stride>(
                    input + input_base + c * RP::src_col_stride,
                    output + output_base + c);
        } else {
            Data data;
            GM<1> src(input + input_base);
            MGATHER(data, src, indices);
            GM<RP::dst_row_stride> dst(output + output_base);
            TSTORE(dst, data);
        }
    }

    __attribute__((always_inline)) void copy2(std::uint16_t* input,
                                               std::uint16_t* output,
                                               unsigned in0, unsigned out0,
                                               unsigned in1, unsigned out1) {
        if constexpr (!useGather) {
            copy1(input, output, in0, out0);
            copy1(input, output, in1, out1);
        } else {
            Data data0, data1;
            GM<1> src0(input + in0), src1(input + in1);
            MGATHER(data0, src0, indices);
            MGATHER(data1, src1, indices);
            GM<RP::dst_row_stride> dst0(output + out0), dst1(output + out1);
            TSTORE(dst0, data0);
            TSTORE(dst1, data1);
        }
    }
};

template<class P, unsigned Rows>
__attribute__((always_inline)) inline void run_rect_row_block(
    std::uint16_t* input, std::uint16_t* output, unsigned row_base) {
    using RP = RectPlan<P>;
    constexpr unsigned full_cols = RP::cols / 32;
    constexpr unsigned full_tasks = RP::outer_count * full_cols;
    if constexpr (full_cols != 0) {
        RectTransfer<P, Rows, 32> transfer;
        unsigned task = 0;
        for (; task + 1 < full_tasks; task += 2) {
            unsigned in0, out0, in1, out1;
            transfer.task_bases(task, row_base, in0, out0);
            transfer.task_bases(task + 1, row_base, in1, out1);
            transfer.copy2(input, output, in0, out0, in1, out1);
        }
        if (task < full_tasks) {
            unsigned in0, out0;
            transfer.task_bases(task, row_base, in0, out0);
            transfer.copy1(input, output, in0, out0);
        }
    }
    if constexpr (RP::cols % 32 != 0) {
        constexpr unsigned TailCols = RP::cols % 32;
        RectTransfer<P, Rows, TailCols> transfer;
        unsigned outer = 0;
        for (; outer + 1 < RP::outer_count; outer += 2) {
            unsigned in0, out0, in1, out1;
            transfer.tail_bases(outer, row_base, in0, out0);
            transfer.tail_bases(outer + 1, row_base, in1, out1);
            transfer.copy2(input, output, in0, out0, in1, out1);
        }
        if (outer < RP::outer_count) {
            unsigned in0, out0;
            transfer.tail_bases(outer, row_base, in0, out0);
            transfer.copy1(input, output, in0, out0);
        }
    }
}

template<class P, unsigned Rows, unsigned PeCount>
__attribute__((always_inline)) inline void run_rect_row_block_partitioned(
    std::uint16_t* input, std::uint16_t* output, unsigned row_base,
    unsigned block_index, unsigned row_blocks, unsigned pe) {
    using RP = RectPlan<P>;
    constexpr unsigned full_cols = RP::cols / 32;
    constexpr unsigned full_tasks = RP::outer_count * full_cols;
    if constexpr (full_cols != 0) {
        RectTransfer<P, Rows, 32> transfer;
        const unsigned offset = block_index * full_tasks;
        unsigned task = (pe + PeCount - offset % PeCount) % PeCount;
        for (; task < full_tasks;) {
            unsigned in0, out0;
            transfer.task_bases(task, row_base, in0, out0);
            const unsigned next = task + PeCount;
            if (next < full_tasks) {
                unsigned in1, out1;
                transfer.task_bases(next, row_base, in1, out1);
                transfer.copy2(input, output, in0, out0, in1, out1);
                task = next + PeCount;
            } else {
                transfer.copy1(input, output, in0, out0);
                break;
            }
        }
    }
    if constexpr (RP::cols % 32 != 0) {
        constexpr unsigned TailCols = RP::cols % 32;
        RectTransfer<P, Rows, TailCols> transfer;
        const unsigned offset = row_blocks * full_tasks +
                                block_index * RP::outer_count;
        unsigned outer = (pe + PeCount - offset % PeCount) % PeCount;
        for (; outer < RP::outer_count;) {
            unsigned in0, out0;
            transfer.tail_bases(outer, row_base, in0, out0);
            const unsigned next = outer + PeCount;
            if (next < RP::outer_count) {
                unsigned in1, out1;
                transfer.tail_bases(next, row_base, in1, out1);
                transfer.copy2(input, output, in0, out0, in1, out1);
                outer = next + PeCount;
            } else {
                transfer.copy1(input, output, in0, out0);
                break;
            }
        }
    }
}

template<class P, unsigned PeCount>
__attribute__((always_inline)) inline void transpose_gather_rect(
    std::uint16_t* input, std::uint16_t* output, unsigned pe) {
    using RP = RectPlan<P>;
    constexpr unsigned full_row_blocks = RP::rows / 32;
    constexpr unsigned row_blocks = full_row_blocks + (RP::rows % 32 != 0);
    if constexpr (row_blocks >= PeCount) {
        for (unsigned block = pe; block < full_row_blocks; block += PeCount)
            run_rect_row_block<P, 32>(input, output, block * 32);
        if constexpr (RP::rows % 32 != 0) {
            // Column-copy tails cost far more per element than full gathers.
            // Spread their independent outer/column tasks over every PE.
            run_rect_row_block_partitioned<P, RP::rows % 32, PeCount>(
                input, output, full_row_blocks * 32, full_row_blocks,
                row_blocks, pe);
        }
    } else {
        for (unsigned block = 0; block < full_row_blocks; ++block)
            run_rect_row_block_partitioned<P, 32, PeCount>(
                input, output, block * 32, block, row_blocks, pe);
        if constexpr (RP::rows % 32 != 0)
            run_rect_row_block_partitioned<P, RP::rows % 32, PeCount>(
                input, output, full_row_blocks * 32, full_row_blocks,
                row_blocks, pe);
    }
}

template<class P, unsigned TileElements, unsigned Start, bool Done = (Start == P::plane_elements)>
struct Transfers;

template<class P, unsigned TileElements, unsigned Start>
struct Transfers<P, TileElements, Start, false> {
    static constexpr unsigned remaining = P::plane_elements - Start;
    static constexpr unsigned n = remaining < TileElements ? remaining : TileElements;
    static constexpr unsigned rows = n < 32 ? n : 32;
    static constexpr unsigned cols = n < 32 ? 1 : n / 32;
    static constexpr unsigned used = rows * cols;
    static constexpr unsigned pitch = (cols + 1) / 2 * 2;
    using Index = pto::Tile<pto::Location::Vec, std::uint32_t, 32, pitch,
                            pto::BLayout::CubeM32, rows, cols>;
    using Data = pto::Tile<pto::Location::Vec, std::uint16_t, 32, pitch,
                           pto::BLayout::CubeM32, rows, cols>;
    template<class T, unsigned Stride> using GM = pto::GlobalTensor<T,
        pto::Shape<1,1,1,rows,cols>, pto::Stride<1,1,rows*Stride,Stride,1>, pto::Layout::ND>;
    alignas(128) inline static constexpr auto offsets = [] {
        std::array<std::uint32_t, 32 * pitch> a{};
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned c = 0; c < cols; ++c)
                a[r*pitch+c] = P::input_offset(Start+r*cols+c) * sizeof(std::uint16_t);
        return a;
    }();
    Index indices;
    // Each index tile is loaded once per PE and reused for all its planes.
    __attribute__((always_inline)) Transfers() {
        GM<std::uint32_t, pitch> gm(const_cast<std::uint32_t*>(offsets.data()));
        TLOAD(indices, gm);
    }
    Transfers<P, TileElements, Start + used> rest;
    __attribute__((always_inline)) void copy(std::uint16_t* input, std::uint16_t* output) {
        Data data;
        GM<std::uint16_t, cols> src(input), dst(output + Start);
        MGATHER(data, src, indices);
        TSTORE(dst, data);
        rest.copy(input, output);
    }
};
template<class P, unsigned TileElements, unsigned Start>
struct Transfers<P, TileElements, Start, true> {
    __attribute__((always_inline)) void copy(std::uint16_t*, std::uint16_t*) {}
};
} // namespace transpose_plane_detail

// Out-of-place U16 transpose, rank 1..5. Shape/Perm are constexpr std::arrays.
// TileElements=512 means 1 KiB data / 2 KiB index for full tiles. No SharedTile.
// Up to two trailing output axes are combined, with bounded cached indices.
template<auto Shape, auto Perm, unsigned TileElements = 512,
         unsigned MaxPlaneElements = 4096, unsigned PeCount = kDefaultPeCount>
__attribute__((always_inline)) inline void transpose_gather_plane(std::uint16_t* input, std::uint16_t* output,
                                   unsigned pe = get_thread_idx()) {
    static_assert(TileElements >= 32 && TileElements <= 512 && TileElements % 32 == 0);
    static_assert(PeCount > 0);
    using P = transpose_plane_detail::Plan<Shape, Perm, MaxPlaneElements>;
    using RP = transpose_plane_detail::RectPlan<P>;
    if constexpr (RP::available) {
        if (pe >= PeCount) return;
        transpose_plane_detail::transpose_gather_rect<P, PeCount>(input, output, pe);
        return;
    }
    constexpr unsigned planes = P::count / P::plane_elements;
    if (pe >= PeCount || pe >= planes) return;
    transpose_plane_detail::Transfers<P, TileElements, 0> transfers;
    for (unsigned plane = pe; plane < planes; plane += PeCount) {
        const unsigned dst = plane * P::plane_elements;
        transfers.copy(input + P::input_offset(dst), output + dst);
    }
}
} // namespace supernpu::multi_thread
