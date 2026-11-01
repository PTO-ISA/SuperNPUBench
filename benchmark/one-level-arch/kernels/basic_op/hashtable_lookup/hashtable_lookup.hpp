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
// dst = rotl32(x, r) = (x << r) | (x >>(logical) (32 - r))   (in-place safe)
template <typename TileU32>
inline void tileRotl32(TileU32& dst, TileU32& x, unsigned r, TileU32& tA, TileU32& tB) {
    TSHLS(tA, x, r);
    TSHRS(tB, x, 32u - r);
    TOR(dst, tA, tB);
}

// One MurmurHash3 mixing round for a 32-bit block: folds `block` into `h`.
template <typename TileU32>
inline void murmurRound(TileU32& h, TileU32& block, TileU32& k, TileU32& tA, TileU32& tB) {
    // Write the first mixing result directly; no copy of block is needed.
    TMULS(k, block, c1);
    tileRotl32(k, k, rot_c1, tA, tB);
    TMULS(k, k, c2);
    TXOR(h, h, k);
    tileRotl32(h, h, rot_c2, tA, tB);
    TMULS(h, h, 5u);
    TADDS(h, h, c3);
}

// Unsigned modulo of a full-range uint32 tile by a (compile-time-known) modulus,
// using only signed-int32 TREM (the ISA has no unsigned rem). Identity:
//   h % m == (2 * ((h >> 1) % m) + (h & 1)) % m
// (h>>1) < 2^31 so it is a non-negative int32; 2*(m-1)+1 < 2^31 so the second
// operand stays non-negative -> signed TREM yields the correct unsigned result.
template <int kTileRows, int kTileCols, uint32_t kMod>
void uModU32(typename HashFindTypes<kTileRows, kTileCols>::TileI32& dst,
             typename HashFindTypes<kTileRows, kTileCols>::TileU32& h)
{
    using Types  = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileI32 = typename Types::TileI32;

    TileU32 halfU, bitU, oneU;
    TileI32 halfI, bitI, capI, tmp;

    TSHRS(halfU, h, 1u);                 // h >> 1   (0 .. 2^31-1)
    TEXPANDS(oneU, 1u);
    TAND(bitU, h, oneU);                // h & 1

    TCVT(halfI, halfU);               // -> non-negative int32
    TCVT(bitI, bitU);
    TEXPANDS(capI, static_cast<int32_t>(kMod));

    TREM(tmp, halfI, capI);            // (h>>1) % m
    TMULS(tmp, tmp, 2);                // 2 * r1
    TADD(tmp, tmp, bitI);              // 2*r1 + (h&1)
    TREM(dst, tmp, capI);             // == h % m
}

// Compute per-key probe byte-offset = (MurmurHash3(key) % kCap) * sizeof(TableEntry)
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

    // block0 = low 32 bits, block1 = high 32 bits of each 8-byte key
    TileU32 block0, block1;
    TCVT(block0, queryKeys);          // integer conversion truncates to low 32
    TSHRS(keyHi, queryKeys, 32u);       // low32 of shifted value is original high32
    TCVT(block1, keyHi);

    TEXPANDS(h, DEFAULT_HASH_SEED);
    murmurRound(h, block0, k, tA, tB);
    murmurRound(h, block1, k, tA, tB);

    // finalize: h ^= len(=8); h ^= h>>16; h *= ..; h ^= h>>13; h *= ..; h ^= h>>16
    TileU32 lenTile;
    TEXPANDS(lenTile, 8u);
    TXOR(h, h, lenTile);
    TSHRS(tmp, h, 16u); TXOR(h, h, tmp);
    TMULS(h, h, 0x85ebca6bu);
    TSHRS(tmp, h, 13u); TXOR(h, h, tmp);
    TMULS(h, h, 0xc2b2ae35u);
    TSHRS(tmp, h, 16u); TXOR(h, h, tmp);

    // slot = h % kCap (unsigned) ; byte offset = slot * sizeof(TableEntry)
    uModU32<kTileRows, kTileCols, static_cast<uint32_t>(kCap)>(probeOff, h);
    TMULS(probeOff, probeOff, static_cast<int32_t>(sizeof(TableEntry)));
}


template <uint32_t Capacity>
__attribute__((always_inline)) inline void probe_round(const TableEntry *__restrict table,
    ElementTile<uint64_t, 32> &keys_in, ElementTile<int32_t, 32> &offsets_in,
    ElementTile<int32_t, 32> &result_in, ElementTile<uint32_t, 32> &active_in,
    unsigned valid) {
    // Views update the caller tiles directly; no copy-in/copy-out staging.
    auto &key = TPARTELEMENT(keys_in);
    auto &off = TPARTELEMENT(offsets_in);
    auto &value = TPARTELEMENT(result_in);
    auto &live = TPARTELEMENT(active_in);

#pragma pto element for
    for (unsigned e = 0; e < 32; ++e) {
        unsigned next_live = live[e];
        int32_t next_value = value[e];
        int32_t next_offset = off[e];
        if (e < valid && next_live != 0) {
            const unsigned slot = static_cast<unsigned>(next_offset) >> 4;
            const uint64_t found = static_cast<uint64_t>(table[slot].key);
            if (found == UINT64_MAX) {
                next_live = 0;
            } else if (found == key[e]) {
                next_value = table[slot].value;
                next_live = 0;
            } else {
                next_offset = next_offset + 16;
                if (next_offset == Capacity * 16) next_offset = 0;
            }
        }
        // Single write per view/lane preserves the compiler typed-view contract.
        live[e] = next_live;
        value[e] = next_value;
        off[e] = next_offset;
    }
}

// M32 32x1 predicates occupy bits [31:0], one bit per logical row.
// Ignore padded lanes: active is initialized to one even past the query tail.
inline bool anyActive(ElementTile<uint32_t, 32>& active, unsigned valid) {
    const uint64_t valid_mask = (uint64_t{1} << valid) - 1;
    return (TCMPS<CmpMode::NE>(active, uint32_t{0}) & valid_mask) != 0;
}

// Read-only table, 64-bit keys, MurmurHash3(seed=0), linear probing.
// -1 is reserved as the empty key and cannot be inserted. No tombstones.
// Capacity is bounded so byte offsets fit signed int32 and unsigned modulo
// via two signed TREM operations cannot overflow. Outputs must not alias inputs.
template <uint32_t Capacity>
void lookup(const TableEntry *__restrict table,
            const int64_t *__restrict queries, int32_t *__restrict output,
            unsigned count) {
    static_assert(Capacity > 0 && Capacity <= 0x7fffffffu / sizeof(TableEntry));
    using U32 = ElementTile<uint32_t, 32>;
    using I32 = ElementTile<int32_t, 32>;
    using U64 = ElementTile<uint64_t, 32>;
    for (std::size_t base = 0; base < count; base += 32) {
        const unsigned valid = count - base < 32 ? count - base : 32;
        U64 keys;
        I32 offsets, result;
        U32 active;
        // Preserve the signed public key API; hash and compare its U64 bit pattern.
        TLOAD(keys, reinterpret_cast<const uint64_t *>(queries + base), valid);
        computeProbeOffsets<32, 1, Capacity>(offsets, keys);
        TEXPANDS(result, kNotFound);
        TEXPANDS(active, 1u);
        // Compare Tile state directly into a GPR; no GM roundtrip is needed
        // for the scalar loop condition. Mask out inactive query-tail lanes.
        unsigned probes = 0;
        bool searching = true;
        while (searching && probes < Capacity) {
            probe_round<Capacity>(table, keys, offsets, result, active, valid);
            ++probes;
            searching = anyActive(active, valid);
        }
        // Capacity is only a safety bound for a missing key in a full table;
        // ordinary batches stop after their longest actual probe chain.
        // Same dynamic-row M322ND contract as TSTORE_CUBE. ElementTile's
        // typed carrier requires a full valid shape in the current C++ API,
        // so pass the store's runtime row bound without converting the tile.
        int32_t *dst = output + base;
        asm volatile(
            "BSTART.TLSU TSTORE, S32\n"
            "B.DATR M322ND.normal, Null\n"
            "B.DIM zero, 1, ->lb0\n"
            "B.DIM %[rows], 0, ->lb1\n"
            "B.IOT %[src], mask=1111, last\n"
            "B.IOR [%[base],%[stride]], []\n"
            :
            : [src] "Tr"(result.data()), [rows] "r"(valid),
              [base] "r"(dst), [stride] "r"(sizeof(int32_t))
            : "memory");
    }
}
} // namespace supernpu::hash_lookup

namespace supernpu::multi_thread {
// Contiguous, disjoint PE slices; count need not be divisible by 4 or 32.
template <uint32_t Capacity, uint32_t PeCount = 4>
void hashtable_lookup(const hash_lookup::TableEntry *table, const int64_t *queries,
                      int32_t *output, unsigned count) {
    static_assert(PeCount > 0);
    const unsigned tid = get_thread_idx();
    if (tid >= PeCount) return;
    const unsigned begin = static_cast<uint64_t>(count) * tid / PeCount;
    const unsigned end = static_cast<uint64_t>(count) * (tid + 1) / PeCount;
    hash_lookup::lookup<Capacity>(table, queries + begin, output + begin, end - begin);
}
} // namespace supernpu::multi_thread
