#pragma once
#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstddef>

namespace supernpu::hash_lookup {
using namespace pto;
struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};
static_assert(sizeof(TableEntry) == 16 && offsetof(TableEntry, value) == 8);
inline constexpr int32_t kNotFound = -1;
inline constexpr uint32_t DEFAULT_HASH_SEED = 0, c1 = 0xcc9e2d51u,
    c2 = 0x1b873593u, c3 = 0xe6546b64u, rot_c1 = 15, rot_c2 = 13;
template <int R, int C> struct HashFindTypes {
    static_assert(R * C == 32);
    using TileU32 = ElementTile<uint32_t, 32>;
    using TileI32 = ElementTile<int32_t, 32>;
    using TileU64 = ElementTile<uint64_t, 32>;
};
template <typename TileU32>
inline void tileRotl32(TileU32& dst, TileU32& x, unsigned r, TileU32& tA, TileU32& tB) {
    TSHLS(tA, x, r);
    TSHRS(tB, x, 32u - r);
    TOR(dst, tA, tB);
}
template <typename TileU32>
inline void murmurRound(TileU32& h, TileU32& block, TileU32& k, TileU32& tA, TileU32& tB) {
    TMULS(k, block, c1);
    tileRotl32(k, k, rot_c1, tA, tB);
    TMULS(k, k, c2);
    TXOR(h, h, k);
    tileRotl32(h, h, rot_c2, tA, tB);
    TMULS(h, h, 5u);
    TADDS(h, h, c3);
}
template <int kTileRows, int kTileCols, uint32_t kMod>
void uModU32(typename HashFindTypes<kTileRows, kTileCols>::TileI32& dst,
             typename HashFindTypes<kTileRows, kTileCols>::TileU32& h)
{
    using Types  = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileI32 = typename Types::TileI32;

    TileU32 halfU, bitU, oneU;
    TileI32 halfI, bitI, capI, tmp;

    TSHRS(halfU, h, 1u);
    TEXPANDS(oneU, 1u);
    TAND(bitU, h, oneU);

    TCVT(halfI, halfU);
    TCVT(bitI, bitU);
    TEXPANDS(capI, static_cast<int32_t>(kMod));

    TREM(tmp, halfI, capI);
    TMULS(tmp, tmp, 2);
    TADD(tmp, tmp, bitI);
    TREM(dst, tmp, capI);
}
template <int kTileRows, int kTileCols, int kCap>
__attribute__((always_inline)) inline
void computeProbeOffsets(typename HashFindTypes<kTileRows, kTileCols>::TileI32& probeOff,
                         typename HashFindTypes<kTileRows, kTileCols>::TileU64& queryKeys)
{
    using Types  = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileU64 = typename Types::TileU64;

    TileU32 h, k, tA, tB, tmp;
    TileU64 keyHi;

    TileU32 block0, block1;
    TCVT(block0, queryKeys);
    TSHRS(keyHi, queryKeys, 32u);
    TCVT(block1, keyHi);

    TEXPANDS(h, DEFAULT_HASH_SEED);
    murmurRound(h, block0, k, tA, tB);
    murmurRound(h, block1, k, tA, tB);

    TileU32 lenTile;
    TEXPANDS(lenTile, 8u);
    TXOR(h, h, lenTile);
    TSHRS(tmp, h, 16u); TXOR(h, h, tmp);
    TMULS(h, h, 0x85ebca6bu);
    TSHRS(tmp, h, 13u); TXOR(h, h, tmp);
    TMULS(h, h, 0xc2b2ae35u);
    TSHRS(tmp, h, 16u); TXOR(h, h, tmp);

    uModU32<kTileRows, kTileCols, static_cast<uint32_t>(kCap)>(probeOff, h);
    TMULS(probeOff, probeOff, static_cast<int32_t>(sizeof(TableEntry)));
}


template<class T>
__attribute__((always_inline)) inline void gather_masked(T &dst, const void *base,
                                                     ElementTile<int32_t,32> &offset, uint64_t exec_mask) {
    asm volatile(
        "BSTART.TLSU MGATHER, %D[dtype]\n"
        "B.DATR CUBE_M32, DTYPE_NONE, Zero, byte0, Eq, RNONE, nosat, 0, 1\n"
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, 32, ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[off], mask=1111, last, ->%[dst]<%Z[size]>\n"
        "B.IOR [%[base],%[mask],zero], ExecMaskPresent\n"
        : [dst] "=Tr"(dst.data())
        : [dtype] "i"(type_traits<typename T::DType>::TypeCode),
          [size] "i"(tile_type_traits<typename T::TileDType>::TilesizeCode),
          [off] "Tr"(offset.data()), [base] "r"(base),
          [mask] "r"(exec_mask) : "memory");
}
__attribute__((always_inline)) inline void select_i32(ElementTile<int32_t,32> &dst,
    uint64_t mask, ElementTile<int32_t,32> &yes, ElementTile<int32_t,32> &no) {
    asm volatile(
        "BSTART.TEPL TSEL, S32\n"
        "B.DATR Zero\n"
        "B.DIM zero, 1, ->lb0\n"
        "B.DIM zero, 32, ->lb1\n"
        "B.DIM zero, 1, ->lb2\n"
        "B.IOT %[yes], %[no], mask=1111, last, ->%[dst]<%Z[size]>\n"
        "B.IOR [%[mask]], []\n"
        : [dst] "=Tr"(dst.data())
        : [yes] "Tr"(yes.data()), [no] "Tr"(no.data()), [mask] "r"(mask),
          [size] "i"(tile_type_traits<ElementTile<int32_t,32>::TileDType>::TilesizeCode));
}
__attribute__((always_inline)) inline uint64_t equal_keys(ElementTile<uint64_t,32> &a,
                                                          ElementTile<uint64_t,32> &b) {
    uint64_t mask;
    asm volatile("BSTART.TEPL TCMP, U64\n"
        "B.DATR Zero, EQ\n"
        "B.DIM zero, 1, ->lb0\nB.DIM zero, 32, ->lb1\nB.DIM zero, 1, ->lb2\n"
        "B.IOT %[a], %[b], mask=1111, last\nB.IOR [], %[mask]\n"
        : [mask] "=r"(mask) : [a] "Tr"(a.data()), [b] "Tr"(b.data()));
    return mask;
}
__attribute__((always_inline)) inline uint64_t empty_keys(ElementTile<uint64_t,32> &a) {
    uint64_t mask;
    asm volatile("BSTART.TEPL TCMPS, U64\n"
        "B.DATR Zero, EQ\n"
        "B.DIM zero, 1, ->lb0\nB.DIM zero, 32, ->lb1\nB.DIM zero, 1, ->lb2\n"
        "B.IOT %[a], mask=1111, last\nB.IOR [%[empty]], %[mask]\n"
        : [mask] "=r"(mask) : [a] "Tr"(a.data()), [empty] "r"(UINT64_MAX));
    return mask;
}

// Four-slot-per-round linear probing over a guarded table.
//
// Table contract: three additional readable 16-byte entries must follow
// slot Capacity-1, mirroring slots 0..2, and Capacity >= 4.  Each round
// probes o..o+3 with four independent masked gathers, so only the chunk
// base needs a wrap check (the derived slot offsets are plain adds).
template<uint32_t Capacity>
void lookup(const TableEntry *__restrict table, const int64_t *__restrict queries,
            int32_t *__restrict output, unsigned count) {
    static_assert(Capacity >= 4 && Capacity <= 0x7fffffffU/16);
    using U64=ElementTile<uint64_t,32>;
    using I32=ElementTile<int32_t,32>;
    constexpr int32_t kTableBytes = static_cast<int32_t>(Capacity) * static_cast<int32_t>(sizeof(TableEntry));
    for(unsigned base=0;base+32<=count;base+=32) {
        U64 keys, found0, found1, found2, found3;
        I32 offsets, off1, off2, off3, result, values, next, wrapped, zero;
        TLOAD(keys,reinterpret_cast<const uint64_t*>(queries+base),32u);
        computeProbeOffsets<32,1,Capacity>(offsets,keys);
        TEXPANDS(result,0); TEXPANDS(zero,0);
        uint64_t active=0xffffffffULL;
        uint64_t matched=0;
        // Guard-table contract: three entries mirroring slots 0..2 follow
        // slot Capacity-1, so the three derived offsets need no per-slot wrap.
        TADDS(off1, offsets, 16);
        TADDS(off2, offsets, 32);
        TADDS(off3, offsets, 48);
        for(unsigned probe=0;active && probe<Capacity;probe+=4) {
            // Primary gather issues immediately; the secondary offset chain
            // overlaps its latency instead of extending the issue path.
            gather_masked(found0,table,offsets,active);
            gather_masked(found1,table,off1,active);
            gather_masked(found2,table,off2,active);
            gather_masked(found3,table,off3,active);
            const uint64_t empty0=empty_keys(found0);
            const uint64_t equal0=equal_keys(found0,keys);
            const uint64_t hit0=active & equal0 & ~empty0;
            select_i32(result,hit0,offsets,result);
            const uint64_t active1=active & ~(empty0|equal0);

            const uint64_t empty1=empty_keys(found1);
            const uint64_t equal1=equal_keys(found1,keys);
            const uint64_t hit1=active1 & equal1 & ~empty1;
            select_i32(result,hit1,off1,result);
            const uint64_t active2=active1 & ~(empty1|equal1);

            const uint64_t empty2=empty_keys(found2);
            const uint64_t equal2=equal_keys(found2,keys);
            const uint64_t hit2=active2 & equal2 & ~empty2;
            select_i32(result,hit2,off2,result);
            const uint64_t active3=active2 & ~(empty2|equal2);

            const uint64_t empty3=empty_keys(found3);
            const uint64_t equal3=equal_keys(found3,keys);
            const uint64_t hit3=active3 & equal3 & ~empty3;
            select_i32(result,hit3,off3,result);
            matched |= hit0 | hit1 | hit2 | hit3;
            active = active3 & ~(empty3|equal3);

            // offsets += four slots (one base wrap); derived slots are plain adds.
            TADDS(next,off3,16);
            const uint64_t wrap=TCMPS<CmpMode::GE>(next,kTableBytes);
            TADDS(wrapped,next,-kTableBytes);
            select_i32(offsets,wrap,wrapped,next);
            TADDS(off1,offsets,16);
            TADDS(off2,offsets,32);
            TADDS(off3,offsets,48);
        }
        // Coalesce all lane hits into one value gather per 32-key batch.
        gather_masked(values,reinterpret_cast<const char*>(table)+8,result,matched);
        TEXPANDS(next,kNotFound);
        select_i32(result,matched,values,next);
        int32_t *dst=output+base;
        asm volatile(
            "BSTART.TLSU TSTORE, S32\n"
            "B.DATR M322ND.normal, Null\n"
            "B.DIM zero, 1, ->lb0\n"
            "B.DIM zero, 32, ->lb1\n"
            "B.IOT %[src], mask=1111, last\n"
            "B.IOR [%[base],%[stride]], []\n"
            : : [src] "Tr"(result.data()), [base] "r"(dst),
                [stride] "r"(sizeof(int32_t)) : "memory");
    }
}
} // namespace supernpu::hash_lookup
namespace supernpu::multi_thread {
template<uint32_t Capacity,uint32_t PeCount=4>
void hashtable_lookup(const hash_lookup::TableEntry *table,const int64_t *queries,
                      int32_t *output,unsigned count) {
    const unsigned tid=get_thread_idx();
    if(tid>=PeCount)return;
    const unsigned begin=uint64_t(count)*tid/PeCount;
    const unsigned end=uint64_t(count)*(tid+1)/PeCount;
    hash_lookup::lookup<Capacity>(table,queries+begin,output+begin,end-begin);
}
}
