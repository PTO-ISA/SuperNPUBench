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

constexpr uint32_t kC1 = 0xcc9e2d51;
constexpr uint32_t kC2 = 0x1b873593;
constexpr uint32_t kC3 = 0xe6546b64;
constexpr uint32_t kFinal1 = 0x85ebca6b;
constexpr uint32_t kFinal2 = 0xc2b2ae35;
constexpr int64_t kEmptyKey = 0x8000000000000000LL;
constexpr int32_t kNotFound = -1;
constexpr int32_t kLanesPerGroup = 64;

// ============================================================================
// SimdGatherProbe — pure SIMT indexed gather for hashtable_lookup_simd linear probing.
//
// NOT tile programming. Uses raw C pointers passed as __in__/__out__ tile
// parameters. The LLVM compiler handles the tile descriptor to pointer
// lowering. Inside the kernel, blkv_get_tile_ptr extracts the raw data ptr.
//
// Each SIMT lane i loads table[probeIdx[i]] using int64_t byte offsets.
// probeIdx is int64_t (byte address offset into table). Both key+val in one kernel.
// ============================================================================

template <typename tile_shape_out_keys, typename tile_shape_out_vals,
          typename tile_shape_idx>
void __vec__ SimdGatherProbe_Vec_RowMajor(
    typename tile_shape_out_keys::TileDType __out__ keyOut,
    typename tile_shape_out_vals::TileDType __out__ valOut,
    TableEntry __in__ *table,
    const typename tile_shape_idx::TileDType __in__ probeIdx)
{
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape_idx::RowStride + i;
    int64_t byte_offset = blkv_get_tile_ptr(probeIdx)[index];
    TableEntry* entry = reinterpret_cast<TableEntry*>(reinterpret_cast<uint8_t*>(const_cast<TableEntry*>(table)) + byte_offset);
    blkv_get_tile_ptr(keyOut)[index] = entry->key;
    blkv_get_tile_ptr(valOut)[index] = entry->value;
}

template <is_tile_data_v tile_shape_out_keys, is_tile_data_v tile_shape_out_vals,
          is_tile_data_v tile_shape_idx>
void SIMD_GATHER_PROBE_Impl(
    tile_shape_out_keys& keyOut, tile_shape_out_vals& valOut,
    TableEntry __in__ *table,
    const tile_shape_idx& probeIdx)
{
    static_assert(tile_shape_out_keys::ValidRow == tile_shape_idx::ValidRow &&
                  tile_shape_out_keys::ValidCol == tile_shape_idx::ValidCol,
                  "Key output shape must match index tile shape");
    static_assert(tile_shape_out_vals::ValidRow == tile_shape_idx::ValidRow &&
                  tile_shape_out_vals::ValidCol == tile_shape_idx::ValidCol,
                  "Val output shape must match index tile shape");
    static_assert(!tile_shape_out_keys::isBoxedLayout &&
                  !tile_shape_out_vals::isBoxedLayout &&
                  !tile_shape_idx::isBoxedLayout,
                  "Fractal layout not supported");
    static_assert(tile_shape_out_keys::isRowMajor &&
                  tile_shape_out_vals::isRowMajor &&
                  tile_shape_idx::isRowMajor,
                  "Only row-major layout supported for SIMD_GATHER_PROBE");
    static constexpr size_t row = tile_shape_idx::ValidRow;
    static constexpr size_t col = tile_shape_idx::ValidCol;

    SimdGatherProbe_Vec_RowMajor
        <tile_shape_out_keys, tile_shape_out_vals, tile_shape_idx>
        <<<col, row, 1>>>(keyOut.data(), valOut.data(), table, probeIdx.data());
}

#define SIMD_GATHER_PROBE(keyOut, valOut, table, probeIdx) \
    SIMD_GATHER_PROBE_Impl(keyOut, valOut, table, probeIdx)

// ============================================================================
// Tile operation primitives
// ============================================================================

template <typename tile_shape>
void __vec__ TShr_Vec_RowMajor(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src,
    const typename tile_shape::TileDType __in__ shift) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape::RowStride + i;
    typename tile_shape::DType shift_val = blkv_get_tile_ptr(shift)[index];
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index] >> shift_val;
}

template <typename tile_shape>
void __vec__ TShl_Vec_RowMajor(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src,
    const typename tile_shape::TileDType __in__ shift) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape::RowStride + i;
    typename tile_shape::DType shift_val = blkv_get_tile_ptr(shift)[index];
    blkv_get_tile_ptr(dst)[index] = blkv_get_tile_ptr(src)[index] << shift_val;
}

template <typename tile_shape>
void __vec__ TXor_Vec_RowMajor(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src0,
    const typename tile_shape::TileDType __in__ src1) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape::RowStride + i;
    blkv_get_tile_ptr(dst)[index] =
        blkv_get_tile_ptr(src0)[index] ^ blkv_get_tile_ptr(src1)[index];
}

template <typename tile_shape_out, typename tile_shape_in>
void __vec__ TCmp_Vec_RowMajor(
    typename tile_shape_out::TileDType __out__ dst,
    const typename tile_shape_in::TileDType __in__ src0,
    const typename tile_shape_in::TileDType __in__ src1) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape_in::RowStride + i;
    typename tile_shape_in::DType a = blkv_get_tile_ptr(src0)[index];
    typename tile_shape_in::DType b = blkv_get_tile_ptr(src1)[index];
    blkv_get_tile_ptr(dst)[index] = (a == b) ? 1 : 0;
}

// ============================================================================
// Tile operation wrapper functions
// ============================================================================

template <typename tile_shape>
void TSHR_Impl(tile_shape &dst, tile_shape &src, tile_shape &shift) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    TShr_Vec_RowMajor<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>
        (dst.data(), src.data(), shift.data());
}

template <typename tile_shape>
void TSHL_Impl(tile_shape &dst, tile_shape &src, tile_shape &shift) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    TShl_Vec_RowMajor<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>
        (dst.data(), src.data(), shift.data());
}

template <typename tile_shape>
void TXOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    TXor_Vec_RowMajor<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>
        (dst.data(), src0.data(), src1.data());
}

template <typename tile_shape>
void TAND_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    TAnd_Vec_RowMajor<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>
        (dst.data(), src0.data(), src1.data());
}

template <typename tile_shape>
void TOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    TOr_Vec_RowMajor<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>
        (dst.data(), src0.data(), src1.data());
}

template <typename tile_shape_out, typename tile_shape_in>
void TCMP_Impl(tile_shape_out &dst, tile_shape_in &src0, tile_shape_in &src1) {
    static_assert(tile_shape_in::ValidRow != -1 && tile_shape_in::ValidCol != -1,
                  "Only static shape supported");
    TCmp_Vec_RowMajor<tile_shape_out, tile_shape_in><<<tile_shape_in::ValidCol, tile_shape_in::ValidRow, 1>>>
        (dst.data(), src0.data(), src1.data());
}

// ============================================================================
// MurmurHash3 helpers (rotate left)
// ============================================================================

template <typename tile_shape>
void __vec__ RotL15(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape::RowStride + i;
    typename tile_shape::DType x = blkv_get_tile_ptr(src)[index];
    blkv_get_tile_ptr(dst)[index] = (x << 15) | (x >> 17);
}

template <typename tile_shape>
void __vec__ RotL13(
    typename tile_shape::TileDType __out__ dst,
    const typename tile_shape::TileDType __in__ src) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t index = j * tile_shape::RowStride + i;
    typename tile_shape::DType x = blkv_get_tile_ptr(src)[index];
    blkv_get_tile_ptr(dst)[index] = (x << 13) | (x >> 19);
}

template <typename tile_shape>
void ROTL15_Impl(tile_shape &dst, tile_shape &src) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    RotL15<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>(dst.data(), src.data());
}

template <typename tile_shape>
void ROTL13_Impl(tile_shape &dst, tile_shape &src) {
    static_assert(tile_shape::ValidRow != -1 && tile_shape::ValidCol != -1,
                  "Only static shape supported");
    RotL13<tile_shape><<<tile_shape::ValidCol, tile_shape::ValidRow, 1>>>(dst.data(), src.data());
}

template <typename tile_shape>
void TANDS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType scalar) {
    tile_shape scalar_tile;
    TEXPANDSCALAR(scalar_tile, scalar);
    TAND_Impl(dst, src, scalar_tile);
}

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
// Step 3: Linear probing using SimdGatherProbe + SIMT compare/advance
// ============================================================================

// SIMT kernel: compare keys, update output, advance probe byte offset,
// and check if all lanes in each group have found their keys (early break).
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
    using TileU32 = typename Types::TileU32;
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
    using TileU16 = typename Types::TileU16;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;

    using BatchShape  = Shape<1, 1, 1, kTileRows, kTileCols>;
    using BatchStride = Stride<1, 1, 1, kTileCols, 1>;

    using KeyGT = GlobalTensor<int64_t, BatchShape, BatchStride>;
    using OutGT = GlobalTensor<int32_t, BatchShape, BatchStride>;

    TileI64 queryKeyTile;
    TileU32 probeIdxTile;   // int64_t byte offsets for SimdGatherProbe
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
