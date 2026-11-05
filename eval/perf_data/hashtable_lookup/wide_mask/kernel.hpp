#pragma once

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::hash_lookup::wide_mask {
using namespace pto;

struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};
static_assert(sizeof(TableEntry) == 16 && offsetof(TableEntry, value) == 8);

inline constexpr int32_t kNotFound = -1;
inline constexpr uint32_t kHashSeed = 0;
inline constexpr uint32_t kMix1 = 0xcc9e2d51u;
inline constexpr uint32_t kMix2 = 0x1b873593u;
inline constexpr uint32_t kMix3 = 0xe6546b64u;

template <unsigned Lanes> struct Types {
    static_assert(Lanes == 32 || Lanes == 64 || Lanes == 128 ||
                  Lanes == 256 || Lanes == 512);
    static_assert(Lanes % 32 == 0);
    using U32 = ElementTile<uint32_t, Lanes>;
    using I32 = ElementTile<int32_t, Lanes>;
    using U64 = ElementTile<uint64_t, Lanes>;
};

template <typename TileU32>
__attribute__((always_inline)) inline void
rotl32(TileU32 &dst, TileU32 &x, unsigned shift,
        TileU32 &left, TileU32 &right) {
    TSHLS(left, x, shift);
    TSHRS(right, x, 32u - shift);
    TOR(dst, left, right);
}

template <typename TileU32>
__attribute__((always_inline)) inline void
murmurRound(TileU32 &hash, TileU32 &block, TileU32 &mixed,
            TileU32 &left, TileU32 &right) {
    TMULS(mixed, block, kMix1);
    rotl32(mixed, mixed, 15u, left, right);
    TMULS(mixed, mixed, kMix2);
    TXOR(hash, hash, mixed);
    rotl32(hash, hash, 13u, left, right);
    TMULS(hash, hash, 5u);
    TADDS(hash, hash, kMix3);
}

template <unsigned Lanes, uint32_t Modulus>
__attribute__((always_inline)) inline void
unsignedModulo(typename Types<Lanes>::I32 &dst,
               typename Types<Lanes>::U32 &value) {
    using U32 = typename Types<Lanes>::U32;
    using I32 = typename Types<Lanes>::I32;
    U32 halfUnsigned, bitUnsigned, one;
    I32 half, bit, modulus, partial;
    TSHRS(halfUnsigned, value, 1u);
    TEXPANDS(one, 1u);
    TAND(bitUnsigned, value, one);
    TCVT(half, halfUnsigned);
    TCVT(bit, bitUnsigned);
    TEXPANDS(modulus, static_cast<int32_t>(Modulus));
    TREM(partial, half, modulus);
    TMULS(partial, partial, 2);
    TADD(partial, partial, bit);
    TREM(dst, partial, modulus);
}

template <unsigned Lanes, uint32_t Capacity>
__attribute__((always_inline)) inline void
computeProbeOffsets(typename Types<Lanes>::I32 &offsets,
                    typename Types<Lanes>::U64 &keys) {
    using U32 = typename Types<Lanes>::U32;
    using U64 = typename Types<Lanes>::U64;
    U32 hash, mixed, left, right, temporary, low, high, length;
    U64 shifted;
    TCVT(low, keys);
    TSHRS(shifted, keys, 32u);
    TCVT(high, shifted);
    TEXPANDS(hash, kHashSeed);
    murmurRound(hash, low, mixed, left, right);
    murmurRound(hash, high, mixed, left, right);
    TEXPANDS(length, 8u);
    TXOR(hash, hash, length);
    TSHRS(temporary, hash, 16u);
    TXOR(hash, hash, temporary);
    TMULS(hash, hash, 0x85ebca6bu);
    TSHRS(temporary, hash, 13u);
    TXOR(hash, hash, temporary);
    TMULS(hash, hash, 0xc2b2ae35u);
    TSHRS(temporary, hash, 16u);
    TXOR(hash, hash, temporary);
    unsignedModulo<Lanes, Capacity>(offsets, hash);
    TMULS(offsets, offsets, static_cast<int32_t>(sizeof(TableEntry)));
}

template <unsigned Lanes, uint32_t Capacity>
__attribute__((always_inline)) inline void
probeRound(const TableEntry *__restrict table,
           typename Types<Lanes>::U64 &__restrict keys,
           typename Types<Lanes>::I32 &__restrict offsets,
           typename Types<Lanes>::I32 &__restrict result,
           typename Types<Lanes>::U32 &__restrict active, unsigned valid) {
    auto &key = TPARTELEMENT(keys);
    auto &offset = TPARTELEMENT(offsets);
    auto &value = TPARTELEMENT(result);
    auto &live = TPARTELEMENT(active);
#pragma pto element for
    for (unsigned element = 0; element < Lanes; ++element) {
        unsigned nextLive = live[element];
        int32_t nextValue = value[element];
        int32_t nextOffset = offset[element];
        if (element < valid && nextLive != 0u) {
            const unsigned slot = static_cast<unsigned>(nextOffset) >> 4u;
            const uint64_t found = static_cast<uint64_t>(table[slot].key);
            if (found == UINT64_MAX) {
                nextLive = 0u;
            } else if (found == key[element]) {
                nextValue = table[slot].value;
                nextLive = 0u;
            } else {
                nextOffset += static_cast<int32_t>(sizeof(TableEntry));
                if (nextOffset == static_cast<int32_t>(Capacity * sizeof(TableEntry)))
                    nextOffset = 0;
            }
        }
        live[element] = nextLive;
        value[element] = nextValue;
        offset[element] = nextOffset;
    }
}

template <unsigned Lanes>
__attribute__((always_inline)) inline bool
anyActive(typename Types<Lanes>::U32 &active) {
    using Reduced = VecTileM32<uint32_t, 32, Lanes / 32, 32, 1>;
    Reduced rowMaximum;
    TROWMAX(rowMaximum, active);
    return TCMPS<CmpMode::NE>(rowMaximum, uint32_t{0}) != 0;
}

template <uint32_t Capacity, unsigned Lanes>
__attribute__((flatten, noinline)) void
lookup(const TableEntry *__restrict table,
       const int64_t *__restrict queries,
       int32_t *__restrict output, unsigned count) {
    static_assert(Capacity > 0 &&
                  Capacity <= 0x7fffffffu / sizeof(TableEntry));
    using U32 = typename Types<Lanes>::U32;
    using I32 = typename Types<Lanes>::I32;
    using U64 = typename Types<Lanes>::U64;
    for (std::size_t base = 0; base < count; base += Lanes) {
        const unsigned valid = count - base < Lanes ? count - base : Lanes;
        // Wide U64 ElementTile transport is a whole-parent operation. The
        // archived 6144/4 fixture gives every PE 1536 keys, divisible by all
        // admitted widths, so no partial U64 parent is formed.
        if (valid != Lanes)
            __builtin_trap();
        U64 keys;
        I32 offsets, result;
        U32 active;
        TLOAD(keys, reinterpret_cast<const uint64_t *>(queries + base), Lanes);
        computeProbeOffsets<Lanes, Capacity>(offsets, keys);
        TEXPANDS(result, kNotFound);
        TEXPANDS(active, 1u);
        unsigned probes = 0;
        while (anyActive<Lanes>(active) && probes < Capacity) {
            probeRound<Lanes, Capacity>(table, keys, offsets, result, active,
                                        valid);
            ++probes;
        }
        global_tensor<int32_t, RowMajor<32, Lanes / 32>> destination(
            output + base);
        TSTORE(destination, result);
    }
}

template <uint32_t Capacity, unsigned Lanes, uint32_t PeCount = 4>
__attribute__((noinline)) void run(const TableEntry *table,
                                   const int64_t *queries, int32_t *output,
                                   unsigned count) {
    static_assert(PeCount == 4);
    const unsigned tid = get_thread_idx();
    if (tid >= PeCount)
        return;
    const unsigned begin = static_cast<uint64_t>(count) * tid / PeCount;
    const unsigned end = static_cast<uint64_t>(count) * (tid + 1u) / PeCount;
    if ((begin % Lanes) != 0u || ((end - begin) % Lanes) != 0u)
        __builtin_trap();
    lookup<Capacity, Lanes>(table, queries + begin, output + begin,
                            end - begin);
}

} // namespace supernpu::hash_lookup::wide_mask
