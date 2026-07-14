#ifndef HASHTABLE_LOOKUP_SIMD_HPP
#define HASHTABLE_LOOKUP_SIMD_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// Pure-SIMD (tile-level) hashtable lookup.
//
// Everything below is expressed with whole-tile operations from the tile-op API
// (TMULS/TADDS/TXOR/TSLL/TSRL/TOR/TAND/TREM for the hash, MGATHER/TCMP/TSELECT for
// the probe). There are NO __vec__ SIMT kernels, no <<<>>> launches, and no per-lane
// control flow (blkv_get_index / blkv_rdadd / data-dependent branches). The probe loop
// is a host loop bounded by kMaxProbe with NO data-dependent early break: a tile-level
// early exit would need to reduce the "still-searching" mask to a host scalar, and the
// tile-op API has no tile->scalar-register reduction (only TROWSUM->tile and
// TCOPYOUT->memory). Since TSELECT keeps already-found lanes immune to later probes,
// running all kMaxProbe iterations is correct; the loop is pure tile ops + a host counter.
// ============================================================================

// MurmurHash3_x86_32 constants
#define DEFAULT_HASH_SEED 0u
#define c1 0xcc9e2d51u
#define c2 0x1b873593u
#define c3 0xe6546b64u
#define rot_c1 15u
#define rot_c2 13u

struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

constexpr int32_t kNotFound = -1;

template <int kTileRows, int kTileCols>
struct HashFindTypes {
    using TileU32 = Tile<Location::Vec, uint32_t, kTileRows, kTileCols, BLayout::RowMajor>;
    using TileI64 = Tile<Location::Vec, int64_t,  kTileRows, kTileCols, BLayout::RowMajor>;
    using TileI32 = Tile<Location::Vec, int32_t,  kTileRows, kTileCols, BLayout::RowMajor>;
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
    TCVT(k, block);
    TMULS(k, k, c1);
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
void computeProbeOffsets(typename HashFindTypes<kTileRows, kTileCols>::TileI32& probeOff,
                         typename HashFindTypes<kTileRows, kTileCols>::TileI64& queryKeys)
{
    using Types  = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileI64 = typename Types::TileI64;

    TileU32 h, k, blk, tA, tB, tmp;
    TileI64 keyHi;

    // block0 = low 32 bits, block1 = high 32 bits of each 8-byte key
    TileU32 block0, block1;
    TCVT(block0, queryKeys);          // static_cast<uint32_t> truncates to low 32
    TSHRS(keyHi, queryKeys, 32u);       // logical >> 32 (unsigned), high 32 -> low
    TCVT(block1, keyHi);

    TEXPANDS(h, DEFAULT_HASH_SEED);
    TCVT(blk, block0);
    murmurRound(h, blk, k, tA, tB);
    TCVT(blk, block1);
    murmurRound(h, blk, k, tA, tB);

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

// ============================================================================
// runHashFind - one batch of kTileRows x kTileCols keys, pure tile ops
// ============================================================================
template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void runHashFind(int32_t __out__ *out,
                 TableEntry __in__ *table,
                 int64_t __in__ *queries)
{
    using Types   = HashFindTypes<kTileRows, kTileCols>;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;

    using BatchShape  = Shape<1, 1, 1, kTileRows, kTileCols>;
    using BatchStride = Stride<1, 1, 1, kTileCols, 1>;
    using KeyGT = GlobalTensor<int64_t, BatchShape, BatchStride>;
    using OutGT = GlobalTensor<int32_t, BatchShape, BatchStride>;

    // Whole table viewed as flat arrays; MGATHER uses byte offsets from the base.
    using TableKeyGT = GlobalTensor<int64_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;
    using TableValGT = GlobalTensor<int32_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;
    TableKeyGT tableKeyGlobal(reinterpret_cast<int64_t*>(table));
    TableValGT tableValGlobal(reinterpret_cast<int32_t*>(reinterpret_cast<char*>(table) + 8));

    TileI64 queryKeyTile;
    TileI32 probeOffTile;
    TileI32 outTile;

    KeyGT key_gt(queries);
    TLOAD(queryKeyTile, key_gt);

    computeProbeOffsets<kTileRows, kTileCols, kCap>(probeOffTile, queryKeyTile);

    TEXPANDS(outTile, kNotFound);

    // TCMP EQ supports int32 only, so compare the 64-bit keys as two int32 halves.
    TileI32 queryLo, queryHi;
    {
        TileI64 qHi;
        TCVT(queryLo, queryKeyTile);          // low 32 bits
        TSHRS(qHi, queryKeyTile, 32u);
        TCVT(queryHi, qHi);                    // high 32 bits
    }

    TileI64 tableKeyTile;
    TileI64 tableKeyHi;
    TileI32 tableLo, tableHi;
    TileI32 tableValTile;
    TileI32 matchLo, matchHi, matchMask;
    TileI32 capBytesTile;
    TEXPANDS(capBytesTile, static_cast<int32_t>(kCap) * static_cast<int32_t>(sizeof(TableEntry)));

    for (int probe = 0; probe < kMaxProbe; probe++) {
        MGATHER(tableKeyTile, tableKeyGlobal, probeOffTile);
        MGATHER(tableValTile, tableValGlobal, probeOffTile);
        // 64-bit key equality = (low32 == low32) && (high32 == high32)
        TCVT(tableLo, tableKeyTile);
        TSHRS(tableKeyHi, tableKeyTile, 32u);
        TCVT(tableHi, tableKeyHi);
        TCMP(matchLo, queryLo, tableLo);
        TCMP(matchHi, queryHi, tableHi);
        TAND(matchMask, matchLo, matchHi);
        TSEL(outTile, matchMask, tableValTile);   // match ? value : keep

        // NOTE: no early break. A data-dependent early exit would require reducing the
        // "still-searching" mask to a host scalar (TROWSUMEXPAND -> TCOPYOUT -> scalar
        // load), i.e. a tile->memory->scalar round-trip every iteration. The tile-op API
        // has no tile->scalar-register reduction, so that round-trip is the only option
        // and it is exactly the fragile path we avoid. TSELECT already makes already-found
        // lanes immune to further probes, so running all kMaxProbe iterations is correct;
        // we simply trade the early-exit speedup for a pure-tile loop body.

        // advance: probeOff = (probeOff + sizeof(TableEntry)) % (kCap * sizeof(TableEntry))
        TADDS(probeOffTile, probeOffTile, static_cast<int32_t>(sizeof(TableEntry)));
        TREM(probeOffTile, probeOffTile, capBytesTile);
    }

    OutGT outGlobal(out);
    TSTORE(outGlobal, outTile);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t *out, TableEntry* table, int64_t* queries)
{
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(out, table, queries);
}

#endif // HASHTABLE_LOOKUP_SIMD_HPP
