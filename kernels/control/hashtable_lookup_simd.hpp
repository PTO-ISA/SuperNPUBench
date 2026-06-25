#ifndef HASHTABLE_LOOKUP_SIMD_HPP
#define HASHTABLE_LOOKUP_SIMD_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

// ============================================================================
// MurmurHash3 constants
// ============================================================================

#define DEFAULT_HASH_SEED 0u
#define c1 0xcc9e2d51u
#define c2 0x1b873593u
#define c3 0xe6546b64u
#define rot_c1 15u
#define rot_c2 13u

// ============================================================================
// Data structures
// ============================================================================

struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

// ============================================================================
// Kernel constants
// ============================================================================

constexpr int32_t kNotFound = -1;
constexpr int32_t kLanesPerGroup = 64;

// ============================================================================
// Tile type aliases
// ============================================================================

template <int kTileRows, int kTileCols>
struct HashFindTypes {
    using TileU32 = Tile<Location::Vec, uint32_t, kTileRows, kTileCols, BLayout::RowMajor>;
    using TileU16 = Tile<Location::Vec, uint16_t, kTileRows, kTileCols, BLayout::RowMajor>;
    using TileI64 = Tile<Location::Vec, int64_t,  kTileRows, kTileCols, BLayout::RowMajor>;
    using TileI32 = Tile<Location::Vec, int32_t,  kTileRows, kTileCols, BLayout::RowMajor>;
};

// ============================================================================
// Step 2: Compute MurmurHash3 and initial byte offset (writes int64_t)
// ============================================================================
template <typename key_shape, typename probe_idx_shape>
void __vec__ compute_hash_vec_impl(
    typename key_shape::TileDType __in__ keys,
    typename probe_idx_shape::TileDType __out__ probe_idxs,
    uint32_t kCap) {

    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t idx = j * key_shape::RowStride + i;

    __vbuf__ typename key_shape::DType* keys_ptr = blkv_get_tile_ptr(keys);

    uint32_t len = 8u;
    __vbuf__ uint32_t* blocks = (__vbuf__ uint32_t*)(keys_ptr + idx);

    uint32_t h = DEFAULT_HASH_SEED;
    for (int32_t jj = 0; jj < 2; jj++)
    {
        uint32_t k1 = blocks[jj];
        k1 *= c1;
        k1 = (k1 << rot_c1) | (k1 >> (32u - rot_c1));
        k1 *= c2;
        h ^= k1;
        h = (h << rot_c2) | (h >> (32u - rot_c2));
        h = h * 5u + c3;
    }

    // Finalize hash.
    h ^= len;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;

    // Store as int64_t byte offset: (slot_index * sizeof(TableEntry))
    __vbuf__ typename probe_idx_shape::DType* probe_idxs_ptr = blkv_get_tile_ptr(probe_idxs);
    probe_idxs_ptr[idx] = (h % kCap) * 16u;
}

template <typename key_shape, typename probe_idx_shape>
void compute_hash_vec(key_shape& keys, probe_idx_shape& probe_idx, uint32_t kCap)
{
    compute_hash_vec_impl<key_shape, probe_idx_shape><<<key_shape::ValidCol, key_shape::ValidRow, 1>>>(keys.data(), probe_idx.data(), kCap);
}

// ============================================================================
// Step 3: Linear probing — compare keys, update output, advance probe offset
// ============================================================================

template <typename key_shape, typename value_shape, typename idx_shape,
          typename out_shape, typename count_shape>
void __vec__ probe_step_impl(
    typename key_shape::TileDType __in__ query_keys,
    typename key_shape::TileDType __in__ table_keys,
    typename value_shape::TileDType __in__ table_values,
    typename idx_shape::TileDType __in__ probe_idx_in,
    typename idx_shape::TileDType __out__ probe_idx_out,
    typename out_shape::TileDType __in__ out_in,
    typename out_shape::TileDType __out__ out_out,
    typename count_shape::TileDType __out__ groupCount,
    uint32_t kCap,
    int32_t not_found_val)
{
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t idx = j * key_shape::RowStride + i;

    __vbuf__ int64_t* keys_ptr       = (__vbuf__ int64_t*)blkv_get_tile_ptr(query_keys);
    __vbuf__ int64_t* tkeys_ptr      = (__vbuf__ int64_t*)blkv_get_tile_ptr(table_keys);
    __vbuf__ int32_t* tvals_ptr      = (__vbuf__ int32_t*)blkv_get_tile_ptr(table_values);
    __vbuf__ uint32_t* pidx_in_ptr    = (__vbuf__ uint32_t*)blkv_get_tile_ptr(probe_idx_in);
    __vbuf__ uint32_t* pidx_out_ptr   = (__vbuf__ uint32_t*)blkv_get_tile_ptr(probe_idx_out);
    __vbuf__ int32_t* out_in_ptr     = (__vbuf__ int32_t*)blkv_get_tile_ptr(out_in);
    __vbuf__ int32_t* out_out_ptr    = (__vbuf__ int32_t*)blkv_get_tile_ptr(out_out);

    // Compare and update output
    bool match = (keys_ptr[idx] == tkeys_ptr[idx]);
    out_out_ptr[idx] = match ? tvals_ptr[idx] : out_in_ptr[idx];
    // Advance: add one entry size (16 bytes), wrap around
    uint32_t next_offset = pidx_in_ptr[idx] + 16;
    if (next_offset >= static_cast<int64_t>(kCap) * 16) next_offset = 0;
    pidx_out_ptr[idx] = next_offset;

    // Early break: blkv_rdadd to check if all lanes in group found their keys
    // Use inverted flag: not_found=1 if still searching, 0 if found
    // blkv_rdadd result == 0 means all lanes found
    int64_t not_found = (out_out_ptr[idx] == not_found_val) ? 1 : 0;
    int64_t sum = blkv_rdadd(not_found);
    // Only lane 0 of each 64-lane group writes the per-group sum
    if ((i & 63) == 0) {
        __vbuf__ int64_t* count_ptr = blkv_get_tile_ptr(groupCount);
        size_t group_idx = (j * key_shape::ValidCol + i) / kLanesPerGroup;
        count_ptr[group_idx] = sum;
    }
}

template <typename key_shape, typename value_shape, typename idx_shape,
          typename out_shape, typename count_shape>
void probe_step(key_shape& queryKeyTile,
                key_shape& tableKeyTile,
                value_shape& tableValueTile,
                idx_shape& probeIdxTile,
                out_shape& outTile,
                count_shape& countTile,
                uint32_t kCap,
                int32_t not_found_val)
{
    probe_step_impl<key_shape, value_shape, idx_shape, out_shape, count_shape>
        <<<key_shape::ValidCol, key_shape::ValidRow, 1>>>(
            queryKeyTile.data(), tableKeyTile.data(), tableValueTile.data(),
            probeIdxTile.data(), probeIdxTile.data(),
            outTile.data(), outTile.data(),
            countTile.data(),
            kCap, not_found_val);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void linearProbing(typename HashFindTypes<kTileRows, kTileCols>::TileI32& outTile,
                   typename HashFindTypes<kTileRows, kTileCols>::TileU32& probeIdxTile,
                   typename HashFindTypes<kTileRows, kTileCols>::TileI64& queryKeyTile,
                   TableEntry *table)
{
    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;

    using TableKeyGT = GlobalTensor<int64_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;
    using TableValGT = GlobalTensor<int32_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;
    TableKeyGT tableKeyGlobal(reinterpret_cast<int64_t*>(table));
    TableValGT tableValGlobal(reinterpret_cast<int32_t*>(reinterpret_cast<char*>(table) + 8));
    TileI64 tableKeyTile;
    TileI32 tableValueTile;

    // Early break: per-group count of lanes still searching
    constexpr int kNumGroups = (kTileCols + kLanesPerGroup - 1) / kLanesPerGroup;
    using CountTile = Tile<Location::Vec, int64_t, 1, kNumGroups, BLayout::RowMajor>;
    using CountGTShape  = Shape<1,1,1,1,kNumGroups>;
    using CountGTStride = Stride<1,1,1,kNumGroups,1>;
    using CountGT = GlobalTensor<int64_t, CountGTShape, CountGTStride>;

    int64_t found_counts[kNumGroups];
    CountTile countTile;
    CountGT countGT(found_counts);

    for (int probe = 0; probe < kMaxProbe; probe++) {
        MGATHER(tableKeyTile, tableKeyGlobal, probeIdxTile);
        MGATHER(tableValueTile, tableValGlobal, probeIdxTile);
        // Compare, advance, and check if all lanes found
        probe_step(queryKeyTile, tableKeyTile, tableValueTile, probeIdxTile, outTile, countTile, kCap, kNotFound);

        TCOPYOUT(countGT, countTile);

        bool all_done = true;
        for (int g = 0; g < kNumGroups; g++) {
            if (found_counts[g] != 0) { all_done = false; break; }
        }
        if (all_done) break;
    }
}

// ============================================================================
// runHashFind - processes one batch of 256 keys
// ============================================================================

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void runHashFind(int32_t __out__ *out,
                 TableEntry __in__ *table,
                 int64_t __in__ *queries)
{
    // capacity can be non-power-of-two; we use modulo arithmetic

    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;

    using BatchShape  = Shape<1, 1, 1, kTileRows, kTileCols>;
    using BatchStride = Stride<1, 1, 1, kTileCols, 1>;

    using KeyGT = GlobalTensor<int64_t, BatchShape, BatchStride>;
    using OutGT = GlobalTensor<int32_t, BatchShape, BatchStride>;

    TileI64 queryKeyTile;
    TileU32 probeIdxTile;   // int64_t byte offsets for gather
    TileI32 outTile;

    // copy in
    KeyGT key_gt(queries);
    TCOPYIN(queryKeyTile, key_gt);

    // compute hash (writes int64_t byte offsets into probeIdxTile)
    compute_hash_vec(queryKeyTile, probeIdxTile, kCap);

    TEXPANDSCALAR(outTile, kNotFound);
    // linear probing
    linearProbing<kTileRows, kTileCols, kCap, kMaxProbe>(
        outTile, probeIdxTile, queryKeyTile, table);

    // copy out
    OutGT outGlobal(out);
    TCOPYOUT(outGlobal, outTile);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t *out, TableEntry* table, int64_t* queries)
{
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(
        out, table, queries);
}

#endif // HASHTABLE_LOOKUP_SIMD_HPP
