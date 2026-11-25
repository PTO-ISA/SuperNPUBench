#pragma once
#include "basic_op/utils/spmd_partition.hpp"
#include <array>
#include <cstdint>

namespace supernpu::multi_thread {
namespace transpose_detail {
// Strides are in U16 elements; padding is local to the register Tile.
template<unsigned Rows, unsigned Cols, unsigned SrcStride, unsigned DstStride>
inline void copy(std::uint16_t* input, std::uint16_t* output) {
    // Keep odd multi-column tails in power-of-two transfers. TimingSim's
    // CUBE path currently rejects the seven-column descriptor/capacity pair.
    if constexpr (Cols > 1 && (Cols & (Cols - 1)) != 0) {
        constexpr unsigned head = [] {
            unsigned n = 1;
            while (n * 2 < Cols) n *= 2;
            return n;
        }();
        copy<Rows, head, SrcStride, DstStride>(input, output);
        copy<Rows, Cols - head, SrcStride, DstStride>(input + head, output + head);
    } else {
        static_assert(Rows <= 32, "CellReg M16/M32 Tile supports at most 32 rows");
        constexpr bool m16 = Rows <= 16;
        constexpr unsigned cellRows = m16 ? 16 : 32;
        constexpr unsigned cellCols = m16 ? 4 : 2; // 128-byte U16 CELL
        constexpr unsigned padded = (Cols + cellCols - 1) / cellCols * cellCols;
        using Tile = pto::Tile<pto::Location::Vec, std::uint16_t, cellRows,
            padded, m16 ? pto::BLayout::CubeM16 : pto::BLayout::CubeM32, Rows, Cols>;
        using Src = pto::GlobalTensor<std::uint16_t, pto::Shape<1,1,1,Rows,Cols>,
            pto::Stride<1,1,Rows*SrcStride,SrcStride,1>, pto::Layout::ND>;
        using Dst = pto::GlobalTensor<std::uint16_t, pto::Shape<1,1,1,Rows,Cols>,
            pto::Stride<1,1,Rows*DstStride,DstStride,1>, pto::Layout::ND>;
        Tile tile;
        Src src(input);
        Dst dst(output);
        TLOAD(tile, src);
        TSTORE(dst, tile);
    }
}
} // namespace transpose_detail

// Out-of-place U16 transpose, rank 1..5. Shape/Perm are constexpr std::arrays.
// Bounded Tiles, no SharedTile. Each PE owns consecutive output Tile blocks.
// When the input's last axis is preserved, copy up to TileCols contiguous
// elements per row. Otherwise use a single column with an affine row stride.
template<auto Shape, auto Perm, unsigned TileRows = 0, unsigned TileCols = 64,
         unsigned PeCount = kDefaultPeCount>
inline void transpose_tload(std::uint16_t* input, std::uint16_t* output,
                            unsigned pe = get_thread_idx()) {
    constexpr unsigned rank = Shape.size();
    static_assert(rank >= 1 && rank <= 5 && Perm.size() == rank);
    static_assert(TileCols > 0 && PeCount > 0);
    static_assert(TileRows <= 32 && TileCols <= 256);
    static_assert([] {
        std::uint64_t count = 1;
        for (unsigned i = 0; i < rank; ++i) {
            if (Shape[i] == 0 || Perm[i] >= rank) return false;
            for (unsigned j = 0; j < i; ++j)
                if (Perm[i] == Perm[j]) return false;
            if (Shape[i] > 0x3fffffffULL / count) return false;
            count *= Shape[i];
        }
        return true;
    }(), "Invalid shape/permutation or tensor exceeds signed 32-bit byte offsets");
    constexpr auto outShape = [] {
        std::array<unsigned, rank> a{};
        for (unsigned i = 0; i < rank; ++i) a[i] = Shape[Perm[i]];
        return a;
    }();
    constexpr auto inStride = [] {
        std::array<unsigned, rank> a{};
        unsigned stride = 1;
        for (int i = rank - 1; i >= 0; --i) { a[i] = stride; stride *= Shape[i]; }
        return a;
    }();
    constexpr bool contiguous = Perm[rank-1] == rank-1;
    constexpr unsigned rowAxis = contiguous && rank > 1 ? rank-2 : rank-1;
    constexpr unsigned rows = rank == 1 ? 1 : outShape[rowAxis];
    constexpr unsigned cols = contiguous ? outShape[rank-1] : 1;
    constexpr unsigned C = cols < TileCols ? cols : TileCols;
    // Each Local Tile uses one M16/M32 row group. Tail validity is static.
    constexpr unsigned selectedRows = [=] {
        if constexpr (TileRows != 0) return TileRows;
        if constexpr (!contiguous) return 32u;
        return C <= 16 ? 32u : 8u;
    }();
    constexpr unsigned R = rows < selectedRows ? rows : selectedRows;
    constexpr unsigned rowChunks = (rows + R - 1) / R;
    constexpr unsigned colChunks = (cols + C - 1) / C;
    constexpr unsigned tailR = rows % R == 0 ? R : rows % R;
    constexpr unsigned tailC = cols % C == 0 ? C : cols % C;
    constexpr unsigned pitch = inStride[Perm[rowAxis]];
    constexpr unsigned outer = [=] {
        unsigned n = 1;
        for (unsigned i = 0; i < rowAxis; ++i) n *= outShape[i];
        return n;
    }();
    constexpr unsigned blocks = outer * rowChunks * colChunks;
    constexpr unsigned perPe = (blocks + PeCount - 1) / PeCount;
    if (pe >= PeCount) return;
    const unsigned end = (pe + 1) * perPe < blocks ? (pe + 1) * perPe : blocks;
    for (unsigned block = pe * perPe; block < end; ++block) {
        const unsigned col = block % colChunks;
        const unsigned row = block / colChunks % rowChunks;
        const unsigned group = block / (colChunks * rowChunks);
        unsigned q = group;
        unsigned src = row * R * pitch + col * C;
        for (int d = int(rowAxis) - 1; d >= 0; --d) {
            src += (q % outShape[d]) * inStride[Perm[d]];
            q /= outShape[d];
        }
        const unsigned dst = (group * rows + row * R) * cols + col * C;
        // Four statically sized variants cover both row and column tails.
        if (row == rowChunks - 1 && col == colChunks - 1)
            transpose_detail::copy<tailR,tailC,pitch,cols>(input+src,output+dst);
        else if (row == rowChunks - 1)
            transpose_detail::copy<tailR,C,pitch,cols>(input+src,output+dst);
        else if (col == colChunks - 1)
            transpose_detail::copy<R,tailC,pitch,cols>(input+src,output+dst);
        else
            transpose_detail::copy<R,C,pitch,cols>(input+src,output+dst);
    }
}
} // namespace supernpu::multi_thread
