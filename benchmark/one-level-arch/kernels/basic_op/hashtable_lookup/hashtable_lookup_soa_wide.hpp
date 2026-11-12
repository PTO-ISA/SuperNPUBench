#pragma once

#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

// SoA wide-lane linear probing (adaptive two-plus-six).
//
// This is the operator-side home of the "256-lane 2+6" reference kernel that
// was measured at 81,317 cycles (random SoC, mean latency 300, RWDB=48) and
// 77,312 cycles with TAG_WAY=7.  It is NOT a drop-in replacement for the AoS
// entry point in hashtable_lookup.hpp: the table is split into a uint32 key
// array and an int32 value array, the probe runs on wide tiles (Lanes =
// 64/128/256/512), and a batch map schedules whole Lanes-sized batches.
//
// Table contract
//   - keys[] and values[] are SoA arrays with Capacity entries each;
//   - the probe may read up to slot Capacity-1+7 (phase A +1, phase B +6), so
//     both arrays must have at least 7 readable guard entries after
//     Capacity-1; the archived fixture appends 128 (512 B) entries mirroring
//     slots 0..127 and that is the recommended padding;
//   - count must be a multiple of Lanes;
//   - the value gather reads values[hit] with hit < Capacity, the key guard
//     alone is enough for correctness.
//
// API
//   run<Capacity, Lanes, PeCount>(keys, values, queries, output, batchMap,
//                                 batchesPerPe)
//     - batchMap[bi] * Lanes is the query/output offset of batch bi, and PE t
//       owns batchMap[t * batchesPerPe .. (t+1) * batchesPerPe);
//   lookup<Capacity, Lanes>(keys, values, queries, output, count)
//     - sequential batches in query order (no batch map).

namespace supernpu::hash_lookup::soa_wide {
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


template <unsigned Lanes, uint32_t Capacity>
__attribute__((always_inline)) inline void
probeRoundA(const uint32_t *__restrict keys,
            typename Types<Lanes>::U32 &__restrict keyLow,
            typename Types<Lanes>::I32 &__restrict offsets,
            typename Types<Lanes>::I32 &__restrict hitOffset,
            typename Types<Lanes>::U32 &__restrict active, unsigned valid) {
    auto &key = TPARTELEMENT(keyLow);
    auto &offset = TPARTELEMENT(offsets);
    auto &hit = TPARTELEMENT(hitOffset);
    auto &live = TPARTELEMENT(active);
#pragma pto element for
    for (unsigned element = 0; element < Lanes; ++element) {
        const unsigned liveNow0 = live[element];
        int32_t nextOffset = offset[element];
        int32_t hitNow = hit[element];
        const bool active = (liveNow0 != 0u);
        uint32_t f0 = 0, f1 = 0;
        if (active) {
            f0 = keys[static_cast<unsigned>(nextOffset + 0)];
            f1 = keys[static_cast<unsigned>(nextOffset + 1)];
        }
        const bool eq0 = (f0 == key[element]);
        const bool eq1 = (f1 == key[element]);
        const bool hitAny = eq0 || eq1;
        unsigned liveNow = liveNow0;
        if (active) {
            hitNow = hitAny ? ((eq0 ? (nextOffset + 0) : (nextOffset + 1))) : hitNow;
            liveNow = hitAny ? 0u : liveNow0;
        }
        nextOffset += 2;
        live[element] = liveNow;
        offset[element] = nextOffset;
        hit[element] = hitNow;
    }
}

template <unsigned Lanes, uint32_t Capacity>
__attribute__((always_inline)) inline void
probeRoundB(const uint32_t *__restrict keys,
            typename Types<Lanes>::U32 &__restrict keyLow,
            typename Types<Lanes>::I32 &__restrict offsets,
            typename Types<Lanes>::I32 &__restrict hitOffset,
            typename Types<Lanes>::U32 &__restrict active, unsigned valid) {
    auto &key = TPARTELEMENT(keyLow);
    auto &offset = TPARTELEMENT(offsets);
    auto &hit = TPARTELEMENT(hitOffset);
    auto &live = TPARTELEMENT(active);
#pragma pto element for
    for (unsigned element = 0; element < Lanes; ++element) {
        const unsigned liveNow0 = live[element];
        int32_t nextOffset = offset[element];
        int32_t hitNow = hit[element];
        const bool active = (liveNow0 != 0u);
        uint32_t f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0;
        if (active) {
            f0 = keys[static_cast<unsigned>(nextOffset + 0)];
            f1 = keys[static_cast<unsigned>(nextOffset + 1)];
            f2 = keys[static_cast<unsigned>(nextOffset + 2)];
            f3 = keys[static_cast<unsigned>(nextOffset + 3)];
            f4 = keys[static_cast<unsigned>(nextOffset + 4)];
            f5 = keys[static_cast<unsigned>(nextOffset + 5)];
        }
        const bool eq0 = (f0 == key[element]);
        const bool eq1 = (f1 == key[element]);
        const bool eq2 = (f2 == key[element]);
        const bool eq3 = (f3 == key[element]);
        const bool eq4 = (f4 == key[element]);
        const bool eq5 = (f5 == key[element]);
        const bool hitAny = eq0 || eq1 || eq2 || eq3 || eq4 || eq5;
        unsigned liveNow = liveNow0;
        if (active) {
            hitNow = hitAny ? ((eq0 ? (nextOffset + 0) : ((eq1 ? (nextOffset + 1) : ((eq2 ? (nextOffset + 2) : ((eq3 ? (nextOffset + 3) : ((eq4 ? (nextOffset + 4) : (nextOffset + 5))))))))))) : hitNow;
            liveNow = hitAny ? 0u : liveNow0;
        }
        nextOffset += 6;
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
__attribute__((always_inline)) inline void
lookupBatch(const uint32_t *__restrict keyArray,
            const int32_t *__restrict values,
            const int64_t *__restrict queries,
            int32_t *__restrict output, std::size_t base) {
    static_assert(Capacity >= 4 &&
                  Capacity <= 0x7fffffffu / sizeof(TableEntry));
    using U32 = typename Types<Lanes>::U32;
    using I32 = typename Types<Lanes>::I32;
    using U64 = typename Types<Lanes>::U64;
    const unsigned valid = Lanes;
    U64 keys;
    U32 keyLow;
    I32 offsets, hitOffset, result;
    U32 active;
    TLOAD(keys, reinterpret_cast<const uint64_t *>(queries + base), Lanes);
    TCVT(keyLow, keys);
    computeProbeOffsets<Lanes, Capacity>(offsets, keys);
    TEXPANDS(hitOffset, kNotFound);
    TEXPANDS(result, kNotFound);
    TEXPANDS(active, 1u);
    unsigned probes = 0;
    while (anyActive<Lanes>(active) && probes < Capacity) {
        probeRoundA<Lanes, Capacity>(keyArray, keyLow, offsets, hitOffset,
                                     active, valid);
        probes += 2;
        probeRoundB<Lanes, Capacity>(keyArray, keyLow, offsets, hitOffset,
                                     active, valid);
        probes += 6;
    }
    // Deferred value gather: one masked read per completed batch.
    {
        auto &hit = TPARTELEMENT(hitOffset);
        auto &value = TPARTELEMENT(result);
#pragma pto element for
        for (unsigned element = 0; element < Lanes; ++element) {
            if (hit[element] >= 0) {
                value[element] = values[static_cast<unsigned>(hit[element])];
            }
        }
    }
    global_tensor<int32_t, RowMajor<32, Lanes / 32>> destination(output + base);
    TSTORE(destination, result);
}

// Sequential form: batches are the Lanes-sized query blocks in query order.
template <uint32_t Capacity, unsigned Lanes>
__attribute__((flatten, noinline)) void
lookup(const uint32_t *__restrict keyArray,
       const int32_t *__restrict values,
       const int64_t *__restrict queries,
       int32_t *__restrict output, unsigned count) {
    for (std::size_t base = 0; base + Lanes <= count; base += Lanes) {
        lookupBatch<Capacity, Lanes>(keyArray, values, queries, output, base);
    }
}

template <uint32_t Capacity, unsigned Lanes, uint32_t PeCount = 4>
__attribute__((noinline)) void run(const uint32_t *keyArray, const int32_t *values,
                                   const int64_t *queries, int32_t *output,
                                   const unsigned *batchMap,
                                   unsigned batchesPerPe) {
    static_assert(PeCount == 4);
    const unsigned tid = get_thread_idx();
    if (tid >= PeCount)
        return;
    for (unsigned bi = 0; bi < batchesPerPe; ++bi) {
        const std::size_t base =
            static_cast<std::size_t>(batchMap[tid * batchesPerPe + bi]) * Lanes;
        lookupBatch<Capacity, Lanes>(keyArray, values, queries, output, base);
    }
}

} // namespace supernpu::hash_lookup::soa_wide
