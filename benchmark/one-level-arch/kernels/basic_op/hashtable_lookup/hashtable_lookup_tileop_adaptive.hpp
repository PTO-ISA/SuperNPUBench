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
        "B.DATR CUBE_M32, Zero\n"
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

// Adaptive two-plus-N probing over a guarded table.
//
// Tail width is chosen by HASH_ADAPTIVE_TAIL (default 6):
//   phase A: gather slots o, o+1 for every still-active lane;
//   phase B: gather slots o+2 .. o+1+N ONLY for lanes that missed both A slots.
//
// The old stride4 round issued all four gathers under the round-start mask, so
// the ~80% of lanes that resolve in the first two slots still paid for the
// remaining slots.  Here the B gathers carry the post-A mask (activeB), which
// removes those speculative requests, and the whole B group is issued back to
// back before its compares so all of its round trips overlap.
//
// Guard contract: Capacity-1 must be followed by HASH_ADAPTIVE_TAIL + 1
// readable 16-byte entries mirroring slots 0..HASH_ADAPTIVE_TAIL, and
// Capacity >= 4.  Tail 2 -> 3 entries / 48 bytes (same as stride4_guard);
// tail 6 -> 7 entries / 112 bytes.
#ifndef HASH_ADAPTIVE_TAIL
#define HASH_ADAPTIVE_TAIL 6
#endif
static_assert(HASH_ADAPTIVE_TAIL == 2 || HASH_ADAPTIVE_TAIL == 6,
              "HASH_ADAPTIVE_TAIL must be 2 or 6");

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
        U64 found4, found5, found6, found7;
        I32 off4, off5, off6, off7;
        TLOAD(keys,reinterpret_cast<const uint64_t*>(queries+base),32u);
        computeProbeOffsets<32,1,Capacity>(offsets,keys);
        TEXPANDS(result,0); TEXPANDS(zero,0);
        uint64_t active=0xffffffffULL;
        uint64_t matched=0;
        // Guard-table contract: HASH_ADAPTIVE_TAIL+1 entries mirroring slots
        // 0..HASH_ADAPTIVE_TAIL follow slot Capacity-1, so the derived offsets
        // need no per-slot wrap.
        TADDS(off1, offsets, 16);
        TADDS(off2, offsets, 32);
        TADDS(off3, offsets, 48);
        if constexpr (HASH_ADAPTIVE_TAIL == 6) {
            TADDS(off4, offsets, 64);
            TADDS(off5, offsets, 80);
            TADDS(off6, offsets, 96);
            TADDS(off7, offsets, 112);
        }
        for(unsigned probe=0;active && probe<Capacity;probe+=2+HASH_ADAPTIVE_TAIL) {
            // ---- phase A: slots 0 and 1 under the round-start mask ----
            gather_masked(found0,table,offsets,active);
            gather_masked(found1,table,off1,active);
            const uint64_t empty0=empty_keys(found0);
            const uint64_t equal0=equal_keys(found0,keys);
            const uint64_t hit0=active & equal0 & ~empty0;
            select_i32(result,hit0,offsets,result);
            const uint64_t active1=active & ~(empty0|equal0);

            const uint64_t empty1=empty_keys(found1);
            const uint64_t equal1=equal_keys(found1,keys);
            const uint64_t hit1=active1 & equal1 & ~empty1;
            select_i32(result,hit1,off1,result);

            // ---- phase B: only lanes that missed both A slots ----
            // activeB is the post-A mask, so lanes resolved in A issue no
            // request here (this is the whole point of the adaptive split).
            const uint64_t activeB=active1 & ~(empty1|equal1);
            gather_masked(found2,table,off2,activeB);
            gather_masked(found3,table,off3,activeB);
            if constexpr (HASH_ADAPTIVE_TAIL == 6) {
                gather_masked(found4,table,off4,activeB);
                gather_masked(found5,table,off5,activeB);
                gather_masked(found6,table,off6,activeB);
                gather_masked(found7,table,off7,activeB);
            }
            const uint64_t empty2=empty_keys(found2);
            const uint64_t equal2=equal_keys(found2,keys);
            const uint64_t hit2=activeB & equal2 & ~empty2;
            select_i32(result,hit2,off2,result);
            const uint64_t active3=activeB & ~(empty2|equal2);

            const uint64_t empty3=empty_keys(found3);
            const uint64_t equal3=equal_keys(found3,keys);
            const uint64_t hit3=active3 & equal3 & ~empty3;
            select_i32(result,hit3,off3,result);
            matched |= hit0 | hit1 | hit2 | hit3;
            uint64_t active4 = active3 & ~(empty3|equal3);
            uint64_t hit4=0, hit5=0, hit6=0, hit7=0;
            uint64_t empty4=0, equal4=0, empty5=0, equal5=0;
            uint64_t empty6=0, equal6=0, empty7=0, equal7=0;
            if constexpr (HASH_ADAPTIVE_TAIL == 6) {
                empty4=empty_keys(found4);
                equal4=equal_keys(found4,keys);
                hit4=active4 & equal4 & ~empty4;
                select_i32(result,hit4,off4,result);
                const uint64_t active5=active4 & ~(empty4|equal4);

                empty5=empty_keys(found5);
                equal5=equal_keys(found5,keys);
                hit5=active5 & equal5 & ~empty5;
                select_i32(result,hit5,off5,result);
                const uint64_t active6=active5 & ~(empty5|equal5);

                empty6=empty_keys(found6);
                equal6=equal_keys(found6,keys);
                hit6=active6 & equal6 & ~empty6;
                select_i32(result,hit6,off6,result);
                const uint64_t active7=active6 & ~(empty6|equal6);

                empty7=empty_keys(found7);
                equal7=equal_keys(found7,keys);
                hit7=active7 & equal7 & ~empty7;
                select_i32(result,hit7,off7,result);
                active4 = active7 & ~(empty7|equal7);
                matched |= hit4 | hit5 | hit6 | hit7;
            }
            active = active4;

            // base offset += (2+tail) slots with one wrap check; derived slots
            // are plain adds.
            if constexpr (HASH_ADAPTIVE_TAIL == 6) {
                TADDS(next,off7,16);
            } else {
                TADDS(next,off3,16);
            }
            const uint64_t wrap=TCMPS<CmpMode::GE>(next,kTableBytes);
            TADDS(wrapped,next,-kTableBytes);
            select_i32(offsets,wrap,wrapped,next);
            TADDS(off1,offsets,16);
            TADDS(off2,offsets,32);
            TADDS(off3,offsets,48);
            if constexpr (HASH_ADAPTIVE_TAIL == 6) {
                TADDS(off4,offsets,64);
                TADDS(off5,offsets,80);
                TADDS(off6,offsets,96);
                TADDS(off7,offsets,112);
            }
        }
        // Coalesce all lane hits into one value gather per 32-key batch.
        // Unmasked: miss lanes read table[8] harmlessly and TSEL overwrites
        // them with kNotFound, and the model's GPR-mask subset only admits
        // U32/U64 gathers anyway.
        asm volatile(
            "BSTART.TLSU MGATHER, S32\n"
            "B.DATR CUBE_M32, Null\n"
            "B.DIM zero, 1, ->lb0\n"
            "B.DIM zero, 32, ->lb1\n"
            "B.DIM zero, 1, ->lb2\n"
            "B.IOT %[off], mask=1111, last, ->%[dst]<%Z[size]>\n"
            "B.IOR [%[base]], []\n"
            : [dst] "=Tr"(values.data())
            : [size] "i"(tile_type_traits<ElementTile<int32_t,32>::TileDType>::TilesizeCode),
              [off] "Tr"(result.data()),
              [base] "r"(reinterpret_cast<const char*>(table)+8) : "memory");
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
