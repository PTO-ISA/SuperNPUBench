#pragma once
#include "basic_op/utils/spmd_partition.hpp"
#include "basic_op/transpose/transpose_tload.hpp"
#include <array>
#include <cstdint>

namespace supernpu::multi_thread {
// Matched U16 paths: identical output partition and scalar address calculation.
// Gather adds one reusable byte-offset Tile; TLOAD uses the same affine stride.
template <unsigned Case> struct TransposeU16Shape {
    static_assert(Case >= 1 && Case <= 5);
    static constexpr unsigned count = Case == 1 ? 129*22*120 :
        Case == 2 ? 8*4096*3 : Case == 3 ? 1476*32 :
        Case == 4 ? 8*64*4*16*7 : 8*8*64*64;
    static constexpr unsigned rows = Case == 5 ? 1 : Case == 4 ? 8 : 32;
    static constexpr unsigned cols = Case == 5 ? 64 : Case == 4 ? 7 : 1;
    static constexpr unsigned pitch = Case == 1 ? 22*120 : Case == 2 ? 3 :
        Case == 3 ? 32 : Case == 4 ? 4*16*7 : 64;
    static constexpr unsigned row_length = Case == 1 ? 129 : Case == 2 ? 4096 :
        Case == 3 ? 1476 : 0;
    static constexpr unsigned chunks = Case <= 3 ? (row_length+rows-1)/rows : 1;
    static constexpr unsigned blocks = Case <= 3 ? count/row_length*chunks : count/(rows*cols);
    static constexpr unsigned tail = Case <= 3 ? row_length % rows : 0;
    static unsigned src(unsigned block) {
        if constexpr (Case <= 3) {
            const unsigned row = block/chunks, start = block%chunks*rows;
            if constexpr (Case == 1) return start*pitch + row%22*120 + row/22;
            if constexpr (Case == 2) return row/3*4096*3 + start*3 + row%3;
            if constexpr (Case == 3) return start*32 + row;
        } else if constexpr (Case == 4) {
            const unsigned r=block*rows;
            return (((r/4096*64 + r%64)*4 + (r/64)%4)*16 + (r/256)%16)*7;
        } else {
            return ((block/512*8 + block%8)*64 + (block/8)%64)*64;
        }
    }
    static unsigned dst(unsigned block) {
        if constexpr (Case <= 3) return block/chunks*row_length + block%chunks*rows;
        else return block*rows*cols;
    }
};

// The MGATHER API emits a direct M32 selector. Keep both its data and
// byte-offset Tiles in M32; offsets are loaded from ordinary ND memory.
template<unsigned Rows, unsigned Cols, unsigned Pitch, bool Gather,
         unsigned DstPitch = Cols, bool Split = (Cols > 1 && (Cols & (Cols-1)) != 0)>
struct TransposeU16Transfer;

template<unsigned Rows, unsigned Cols, unsigned Pitch, unsigned DstPitch>
struct TransposeU16Transfer<Rows,Cols,Pitch,false,DstPitch,false> {
    void copy(std::uint16_t* input,std::uint16_t* output) {
        transpose_detail::copy<Rows,Cols,Pitch,DstPitch>(input,output);
    }
};

template<unsigned Rows, unsigned Cols, unsigned Pitch, unsigned DstPitch>
struct TransposeU16Transfer<Rows,Cols,Pitch,true,DstPitch,false> {
    static_assert(Rows <= 32);
    static constexpr unsigned padded = (Cols+1)/2*2;
    using DataTile=pto::Tile<pto::Location::Vec,std::uint16_t,32,padded,
                             pto::BLayout::CubeM32,Rows,Cols>;
    using IndexTile=pto::Tile<pto::Location::Vec,std::uint32_t,32,padded,
                              pto::BLayout::CubeM32,Rows,Cols>;
    template<typename T, unsigned Stride>
    using Global=pto::GlobalTensor<T,pto::Shape<1,1,1,Rows,Cols>,
        pto::Stride<1,1,Rows*Stride,Stride,1>,pto::Layout::ND>;
    alignas(128) static constexpr auto offsets=[] {
        std::array<std::uint32_t,32*padded> a{};
        for(unsigned r=0;r<Rows;++r)
            for(unsigned c=0;c<Cols;++c) a[r*padded+c]=(r*Pitch+c)*sizeof(std::uint16_t);
        return a;
    }();
    IndexTile indices;
    TransposeU16Transfer() {
        Global<std::uint32_t,padded> gm(const_cast<std::uint32_t*>(offsets.data()));
        TLOAD(indices,gm);
    }
    void copy(std::uint16_t* input,std::uint16_t* output) {
        DataTile tile;
        Global<std::uint16_t,Pitch> src(input);
        Global<std::uint16_t,DstPitch> dst(output);
        MGATHER(tile,src,indices);
        TSTORE(dst,tile);
    }
};

template<unsigned Rows, unsigned Cols, unsigned Pitch, bool Gather, unsigned DstPitch>
struct TransposeU16Transfer<Rows,Cols,Pitch,Gather,DstPitch,true> {
    static constexpr unsigned head=[] {
        unsigned n=1; while(n*2<Cols) n*=2; return n;
    }();
    TransposeU16Transfer<Rows,head,Pitch,Gather,DstPitch> first;
    TransposeU16Transfer<Rows,Cols-head,Pitch,Gather,DstPitch> rest;
    void copy(std::uint16_t* input,std::uint16_t* output) {
        first.copy(input,output);
        rest.copy(input+head,output+head);
    }
};

template<unsigned Case, bool Gather, bool Contiguous = false>
void transpose_u16_shapes(std::uint16_t* input,std::uint16_t* output,
                           unsigned pe=get_thread_idx()) {
    using S=TransposeU16Shape<Case>;
    if(pe>=kDefaultPeCount) return;
    // Contiguous partitions preserve output order within each PE.
    constexpr unsigned perPe = (S::blocks + kDefaultPeCount - 1) / kDefaultPeCount;
    const unsigned begin = Contiguous ? pe * perPe : pe;
    const unsigned limit = Contiguous ? ((pe + 1) * perPe < S::blocks ?
                                        (pe + 1) * perPe : S::blocks) : S::blocks;
    constexpr unsigned step = Contiguous ? 1 : kDefaultPeCount;
    TransposeU16Transfer<S::rows,S::cols,S::pitch,Gather> full;
    if constexpr (S::tail != 0) {
        TransposeU16Transfer<S::tail,S::cols,S::pitch,Gather> tail;
        for(unsigned b=begin;b<limit;b+=step) {
            if(b%S::chunks==S::chunks-1) tail.copy(input+S::src(b),output+S::dst(b));
            else full.copy(input+S::src(b),output+S::dst(b));
        }
    } else {
        for(unsigned b=begin;b<limit;b+=step)
            full.copy(input+S::src(b),output+S::dst(b));
    }
}
} // namespace supernpu::multi_thread
