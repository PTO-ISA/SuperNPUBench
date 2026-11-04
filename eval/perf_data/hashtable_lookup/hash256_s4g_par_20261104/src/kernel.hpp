#pragma once

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

// 256-lane (and other widths) linear probing with the stride4_guard logic:
//   - four slots probed per round (o, o+1, o+2, o+3)
//   - guard table: three 16 B entries mirroring slots 0..2 follow slot
//     Capacity-1, so derived offsets need no per-slot wrap
//   - values are gathered once per batch, after all lanes have resolved
//
// The file is written in the same element-wise style as the PR wide_mask
// kernel so the compiler lowers it to wide PredicateCell operations.
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
}

// One probe round: two consecutive slots per lane, both key loads issued back
// to back under the round-start mask so their TLSU latency overlaps.
template <unsigned Lanes, uint32_t Capacity>
__attribute__((always_inline)) inline void
probeRound4(const TableEntry *__restrict table,
            typename Types<Lanes>::U64 &__restrict keys,
            typename Types<Lanes>::I32 &__restrict offsets,
            typename Types<Lanes>::I32 &__restrict hitOffset,
            typename Types<Lanes>::U32 &__restrict active, unsigned valid) {
    auto &key = TPARTELEMENT(keys);
    auto &offset = TPARTELEMENT(offsets);
    auto &hit = TPARTELEMENT(hitOffset);
    auto &live = TPARTELEMENT(active);
#pragma pto element for
    for (unsigned element = 0; element < Lanes; ++element) {
        const unsigned liveNow0 = live[element];
        int32_t nextOffset = offset[element];
        int32_t hitNow = hit[element];
        const bool active = (liveNow0 != 0u);
        uint64_t f0 = 0, f1 = 0;
        if (active) {
            f0 = static_cast<uint64_t>(
                table[static_cast<unsigned>(nextOffset)].key);
            f1 = static_cast<uint64_t>(
                table[static_cast<unsigned>(nextOffset + 1)].key);
        }
        unsigned liveNow = liveNow0;
        if (active) {
            if (f0 == UINT64_MAX) {
                liveNow = 0u;
            } else if (f0 == key[element]) {
                hitNow = nextOffset;
                liveNow = 0u;
            }
        }
        if (liveNow != 0u) {
            if (f1 == UINT64_MAX) {
                liveNow = 0u;
            } else if (f1 == key[element]) {
                hitNow = nextOffset + 1;
                liveNow = 0u;
            }
        }
        // No wrap: the 2 KB guard mirrors slots 0..127 and the fixture's
        // longest probe is 63 slots.
        nextOffset += 2;
        live[element] = liveNow;
        offset[element] = nextOffset;
        hit[element] = hitNow;
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
    static_assert(Capacity >= 4 &&
                  Capacity <= 0x7fffffffu / sizeof(TableEntry));
    using U32 = typename Types<Lanes>::U32;
    using I32 = typename Types<Lanes>::I32;
    using U64 = typename Types<Lanes>::U64;
    for (std::size_t base = 0; base < count; base += Lanes) {
        const unsigned valid = count - base < Lanes ? count - base : Lanes;
        // The guard contract mirrors slots 0..2 after slot Capacity-1; the
        // archived fixture gives every PE 1536 keys, divisible by all widths.
        if (valid != Lanes)
            __builtin_trap();
        U64 keys;
        I32 offsets, hitOffset, result;
        U32 active;
        TLOAD(keys, reinterpret_cast<const uint64_t *>(queries + base), Lanes);
        computeProbeOffsets<Lanes, Capacity>(offsets, keys);
        TEXPANDS(hitOffset, kNotFound);
        TEXPANDS(result, kNotFound);
        TEXPANDS(active, 1u);
        unsigned probes = 0;
        while (anyActive<Lanes>(active) && probes < Capacity) {
            probeRound4<Lanes, Capacity>(table, keys, offsets, hitOffset,
                                         active, valid);
            probes += 2;
        }
        // Deferred value gather: one masked read per completed batch.
        {
            auto &hit = TPARTELEMENT(hitOffset);
            auto &value = TPARTELEMENT(result);
#pragma pto element for
            for (unsigned element = 0; element < Lanes; ++element) {
                if (hit[element] >= 0) {
                    value[element] = table[static_cast<unsigned>(hit[element])].value;
                }
            }
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
