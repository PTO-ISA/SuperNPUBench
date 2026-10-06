#pragma once

#include "basic_op/utils/spmd_partition.hpp"

#include <common/pto_tileop.hpp>

#include <cstddef>
#include <cstdint>

namespace supernpu::multi_thread {

struct alignas(16) HashTableEntry {
    std::int64_t key;
    std::int32_t value;
    std::int32_t padding;
};

inline constexpr std::int32_t kHashNotFound = -1;

inline std::uint32_t rotl32(std::uint32_t x, unsigned amount) {
    return (x << amount) | (x >> (32u - amount));
}

inline std::uint32_t murmur_hash_key(std::int64_t key) {
    std::uint64_t bits = static_cast<std::uint64_t>(key);
    std::uint32_t h = 0;
    for (unsigned half = 0; half < 2; ++half) {
        std::uint32_t k = static_cast<std::uint32_t>(bits >> (half * 32));
        k *= 0xcc9e2d51u;
        k = rotl32(k, 15);
        k *= 0x1b873593u;
        h ^= k;
        h = rotl32(h, 13);
        h = h * 5u + 0xe6546b64u;
    }
    h ^= 8u;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

template <typename U32Tile>
inline void tile_rotl32(U32Tile &dst, U32Tile &src, std::uint32_t amount,
                        U32Tile &left, U32Tile &right) {
    TSHLS(left, src, amount);
    TSHRS(right, src, 32u - amount);
    TOR(dst, left, right);
}

template <typename U32Tile>
inline void tile_murmur_round(U32Tile &hash, U32Tile &block,
                              U32Tile &left, U32Tile &right) {
    // `block` is disposable.  Mutating it keeps this round to four live tile
    // registers and avoids backend spills on the one-level target.
    TMULS(block, block, 0xcc9e2d51u);
    tile_rotl32(block, block, 15u, left, right);
    TMULS(block, block, 0x1b873593u);
    TXOR(hash, hash, block);
    tile_rotl32(hash, hash, 13u, left, right);
    TMULS(hash, hash, 5u);
    TADDS(hash, hash, 0xe6546b64u);
}

// SIMD lookup: 256 independent lanes, fixed eight-probe loop, and no
// per-lane control flow.  Capacity is restricted to a power of two so the
// modulo operation is the ISA-exact h & (capacity-1).
template <std::size_t Capacity, std::size_t MaxProbe,
          std::size_t QueryCount, std::size_t TileWidth = 32,
          std::uint32_t PeCount = kDefaultPeCount>
void hashtable_lookup_simd(const HashTableEntry *table,
                           const std::int64_t *queries,
                           std::int32_t *output) {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SIMD hash capacity must be a power of two");
    static_assert(QueryCount % PeCount == 0);
    static_assert((QueryCount / PeCount) % TileWidth == 0);
    static_assert(sizeof(HashTableEntry) == 16);

    // The external num_col=256 block is executed as eight 32-lane ISA tiles.
    // U32 vector tile operations in the one-level ISA use the native M32
    // width; keeping the physical tile at 32 lanes also avoids relying on the
    // legacy two-level API's oversized logical tile convention.
    constexpr std::size_t kQueriesPerPe = QueryCount / PeCount;
    constexpr std::size_t kTilesPerPe = kQueriesPerPe / TileWidth;
    using U32Tile = Tile<Location::Vec, std::uint32_t, 1, TileWidth,
                         BLayout::RowMajor>;
    using I32Tile = Tile<Location::Vec, std::int32_t, 1, TileWidth,
                         BLayout::RowMajor>;
    using PredTile = Tile<Location::Vec, std::uint8_t, 1, TileWidth,
                          BLayout::RowMajor>;
    using QueryWords = global_tensor<std::uint32_t,
                                     RowMajor<1, QueryCount * 2>>;
    using TableWords = global_tensor<std::uint32_t,
                                     RowMajor<1, Capacity * 4>>;
    using TableValues = global_tensor<std::int32_t,
                                      RowMajor<1, Capacity * 4>>;
    using OutputGm = global_tensor<std::int32_t, RowMajor<1, TileWidth>>;

    const std::size_t query_base =
        static_cast<std::size_t>(get_thread_idx()) * kQueriesPerPe;
    TableWords table_low(reinterpret_cast<const std::uint32_t *>(table));
    TableWords table_high(reinterpret_cast<const std::uint32_t *>(table) + 1);
    TableValues table_value(reinterpret_cast<const std::int32_t *>(table) + 2);

    for (std::size_t tile_index = 0; tile_index < kTilesPerPe; ++tile_index) {
        const std::size_t first = query_base + tile_index * TileWidth;
        QueryWords query_low(reinterpret_cast<const std::uint32_t *>(queries + first));
        QueryWords query_high(reinterpret_cast<const std::uint32_t *>(queries + first) + 1);
        U32Tile hash;
        TEXPANDS(hash, 0u);
        {
            U32Tile query_offsets, block, left, right;
            TCI(query_offsets, 0u);
            TMULS(query_offsets, query_offsets, 8u);
            MGATHER(block, query_low, query_offsets);
            tile_murmur_round(hash, block, left, right);
        }
        {
            U32Tile query_offsets, block, left, right;
            TCI(query_offsets, 0u);
            TMULS(query_offsets, query_offsets, 8u);
            MGATHER(block, query_high, query_offsets);
            tile_murmur_round(hash, block, left, right);
        }
        {
            U32Tile tmp;
            TXORS(hash, hash, 8u);
            TSHRS(tmp, hash, 16u); TXOR(hash, hash, tmp);
            TMULS(hash, hash, 0x85ebca6bu);
            TSHRS(tmp, hash, 13u); TXOR(hash, hash, tmp);
            TMULS(hash, hash, 0xc2b2ae35u);
            TSHRS(tmp, hash, 16u); TXOR(hash, hash, tmp);
        }
        U32Tile probe_offsets;
        TANDS(probe_offsets, hash, static_cast<std::uint32_t>(Capacity - 1));
        TSHLS(probe_offsets, probe_offsets, 4u);

        I32Tile result;
        TEXPANDS(result, kHashNotFound);
        // Keep this loop rolled: fully unrolling eight probes increases tile
        // register pressure enough for the backend to spill U32 compare
        // sources through an S64 reload, which violates TCMP's ISA schema.
        #pragma clang loop unroll(disable)
        for (std::size_t probe = 0; probe < MaxProbe; ++probe) {
            // Reload the query halves close to TCMP.  Keeping them live across
            // the hash and probe body triggers a backend tile spill whose
            // reload is incorrectly typed S64; the short lifetime below keeps
            // the generated TCMP sources ISA-compatible U32 tiles.
            U32Tile query_offsets, query_lo, query_hi;
            TCI(query_offsets, 0u);
            TMULS(query_offsets, query_offsets, 8u);
            MGATHER(query_lo, query_low, query_offsets);
            MGATHER(query_hi, query_high, query_offsets);
            U32Tile table_lo, table_hi;
            I32Tile table_val;
            MGATHER(table_lo, table_low, probe_offsets);
            MGATHER(table_hi, table_high, probe_offsets);
            MGATHER(table_val, table_value, probe_offsets);
            PredTile match_lo, match_hi;
            TCMP<CmpMode::EQ>(match_lo, query_lo, table_lo);
            TCMP<CmpMode::EQ>(match_hi, query_hi, table_hi);
            // Predicate tiles are packed logical values in ISA 0.58; a TEPL
            // TAND would turn them into ordinary U8 data and is not a legal
            // TSEL mask.  Two nested selects express low && high while keeping
            // both TCMP results in the logical-predicate domain.
            I32Tile candidate;
            TADDS(candidate, result, 0);
            TSEL(candidate, match_lo, table_val);
            TSEL(result, match_hi, candidate);
            TADDS(probe_offsets, probe_offsets, 16u);
            TANDS(probe_offsets, probe_offsets,
                   static_cast<std::uint32_t>(Capacity * 16 - 1));
        }
        OutputGm dst(output + first);
        TSTORE(dst, result);
    }
}

// SIMT-style scalar lookup.  Each PE owns QueryCount/PeCount logical threads;
// each logical thread follows its own data-dependent probe path and exits as
// soon as its key is found.
template <std::size_t Capacity, std::size_t MaxProbe,
          std::size_t QueryCount,
          std::uint32_t PeCount = kDefaultPeCount>
void hashtable_lookup_simt(const HashTableEntry *table,
                           const std::int64_t *queries,
                           std::int32_t *output) {
    static_assert((Capacity & (Capacity - 1)) == 0);
    static_assert(QueryCount % PeCount == 0);
    constexpr std::size_t kQueriesPerPe = QueryCount / PeCount;
    const std::size_t begin =
        static_cast<std::size_t>(get_thread_idx()) * kQueriesPerPe;
    const std::size_t end = begin + kQueriesPerPe;
    for (std::size_t i = begin; i < end; ++i) {
        const std::int64_t key = queries[i];
        std::size_t slot = murmur_hash_key(key) & (Capacity - 1);
        std::int32_t value = kHashNotFound;
        for (std::size_t probe = 0; probe < MaxProbe; ++probe) {
            if (table[slot].key == key) {
                value = table[slot].value;
                break;
            }
            slot = (slot + 1) & (Capacity - 1);
        }
        output[i] = value;
    }
}

}  // namespace supernpu::multi_thread
