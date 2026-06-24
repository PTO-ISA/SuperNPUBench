#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "template_asm.h"
#include <stdio.h>

// ============================================================================
// Tile operation implementations
// ============================================================================

// ---------------------------------------------------------------------------
// SimdGatherProbe — pure SIMT indexed gather for hashfind linear probing.
//
// NOT tile programming. Uses raw C pointers passed as __in__/__out__ tile
// parameters. The LLVM compiler handles the tile descriptor to pointer
// lowering. Inside the kernel, blkv_get_tile_ptr extracts the raw data ptr.
//
// Each SIMT lane i loads table[probeIdx[i]] using int64_t byte offsets.
// probeIdx is int64_t (byte address offset into table). Both key+val in one kernel.
// ---------------------------------------------------------------------------

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

#define FOR_GFSIM
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
// Constants
// ============================================================================

constexpr uint32_t kCapacity   = 2550000;  // 40800000 / 16
constexpr int32_t  kNum        = 409600;   // 3276800 / 8
constexpr int32_t  kMaxProbe   = 512;
constexpr int32_t  kBatchSize  = 256;      // 16 x 16 tile
constexpr uint32_t kC1 = 0xcc9e2d51;
constexpr uint32_t kC2 = 0x1b873593;
constexpr uint32_t kC3 = 0xe6546b64;
constexpr uint32_t kFinal1 = 0x85ebca6b;
constexpr uint32_t kFinal2 = 0xc2b2ae35;
constexpr int64_t kEmptyKey = 0x8000000000000000LL;
constexpr int32_t kNotFound = -1;

// ============================================================================
// ELF Data layout
// ============================================================================

extern "C" {
    extern const uint8_t _binary_inserted_slot_data_start[];
    extern const uint8_t _binary_inserted_slot_data_end[];
    extern const uint8_t _binary_lookup_keys_data_start[];
    extern const uint8_t _binary_lookup_keys_data_end[];
    extern const uint8_t _binary_lookup_values_data_start[];
    extern const uint8_t _binary_lookup_values_data_end[];
}

static TableEntry* g_hashtable_ro = reinterpret_cast<TableEntry*>(
    const_cast<uint8_t*>(_binary_inserted_slot_data_start));
static int64_t* g_query_keys = reinterpret_cast<int64_t*>(
    const_cast<uint8_t*>(_binary_lookup_keys_data_start));
static int32_t* g_lookup_values = reinterpret_cast<int32_t*>(
    const_cast<uint8_t*>(_binary_lookup_values_data_start));

static int32_t g_output[kNum];

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

    auto* keys_ptr = blkv_get_tile_ptr(keys);

    uint32_t len = 8u;
    uint32_t* blocks = (uint32_t*)(keys_ptr + idx);

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
    auto* probe_idxs_ptr = blkv_get_tile_ptr(probe_idxs);
    probe_idxs_ptr[idx] = static_cast<int64_t>((h % kCap) * 16);
}

template <typename key_shape, typename probe_idx_shape>
void compute_hash_vec(key_shape& keys, probe_idx_shape& probe_idx, uint32_t kCap)
{
    compute_hash_vec_impl<key_shape, probe_idx_shape><<<key_shape::ValidCol, key_shape::ValidRow, 1>>>(keys.data(), probe_idx.data(), kCap);
}

// ============================================================================
// Step 3: Linear probing using SimdGatherProbe + SIMT compare/advance
// ============================================================================

// SIMT kernel: compare keys, update output, advance probe byte offset
template <typename key_shape, typename value_shape, typename idx_shape, typename out_shape>
void __vec__ probe_step_impl(
    typename key_shape::TileDType __in__ query_keys,
    typename key_shape::TileDType __in__ table_keys,
    typename value_shape::TileDType __in__ table_values,
    typename idx_shape::TileDType __in__ probe_idx_in,
    typename idx_shape::TileDType __out__ probe_idx_out,
    typename out_shape::TileDType __in__ out_in,
    typename out_shape::TileDType __out__ out_out,
    uint32_t kCap)
{
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t idx = j * key_shape::RowStride + i;

    int64_t* keys_ptr       = (int64_t*)blkv_get_tile_ptr(query_keys);
    int64_t* tkeys_ptr      = (int64_t*)blkv_get_tile_ptr(table_keys);
    int32_t* tvals_ptr      = (int32_t*)blkv_get_tile_ptr(table_values);
    int64_t* pidx_in_ptr    = (int64_t*)blkv_get_tile_ptr(probe_idx_in);
    int64_t* pidx_out_ptr   = (int64_t*)blkv_get_tile_ptr(probe_idx_out);
    int32_t* out_in_ptr     = (int32_t*)blkv_get_tile_ptr(out_in);
    int32_t* out_out_ptr    = (int32_t*)blkv_get_tile_ptr(out_out);

    out_out_ptr[idx] = (keys_ptr[idx] == tkeys_ptr[idx]) ? tvals_ptr[idx] : out_in_ptr[idx];
    // Advance: add one entry size (16 bytes), wrap around
    int64_t next_offset = pidx_in_ptr[idx] + 16;
    if (next_offset >= static_cast<int64_t>(kCap) * 16) next_offset = 0;
    pidx_out_ptr[idx] = next_offset;
}

template <typename key_shape, typename value_shape, typename idx_shape, typename out_shape>
void probe_step(key_shape& queryKeyTile,
                key_shape& tableKeyTile,
                value_shape& tableValueTile,
                idx_shape& probeIdxTile,
                out_shape& outTile,
                uint32_t kCap)
{
    probe_step_impl<key_shape, value_shape, idx_shape, out_shape>
        <<<key_shape::ValidCol, key_shape::ValidRow, 1>>>(
            queryKeyTile.data(), tableKeyTile.data(), tableValueTile.data(),
            probeIdxTile.data(), probeIdxTile.data(),
            outTile.data(), outTile.data(), kCap);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void linearProbing(typename HashFindTypes<kTileRows, kTileCols>::TileI32& outTile,
                   typename HashFindTypes<kTileRows, kTileCols>::TileI64& probeIdxTile,
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

    // using ProbeGT = GlobalTensor<int64_t, Shape<1,1,1,16,16>, Stride<1,1,1,16,1>>;
    // ProbeGT probe_gt(probe_idx_dump);

    for (int probe = 0; probe < kMaxProbe; probe++) {
        MGATHER(tableKeyTile, tableKeyGlobal, probeIdxTile);
        MGATHER(tableValueTile, tableValGlobal, probeIdxTile);
        // Compare and advance
        probe_step(queryKeyTile, tableKeyTile, tableValueTile, probeIdxTile, outTile, kCap);
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
    TileI64 probeIdxTile;   // int64_t byte offsets for SimdGatherProbe
    TileI32 outTile;

    // copy in
    KeyGT key_gt(queries);
    TLOAD(queryKeyTile, key_gt);

    // compute hash (writes int64_t byte offsets into probeIdxTile)
    compute_hash_vec(queryKeyTile, probeIdxTile, kCap);

    TEXPANDSCALAR(outTile, kNotFound);
    // linear probing
    linearProbing<kTileRows, kTileCols, kCap, kMaxProbe>(
        outTile, probeIdxTile, queryKeyTile, table);

    // copy out
    OutGT outGlobal(out);
    TSTORE(outGlobal, outTile);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t *out, TableEntry* table, int64_t* queries)
{
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(
        out, table, queries);
}

// ============================================================================
// main
// ============================================================================

int main() {
    // Initialize all output to kNotFound
    for (int i = 0; i < kNum; i++) {
        g_output[i] = kNotFound;
    }

    // Process all lookup keys in batches of kBatchSize (256)
    constexpr int32_t num_batches = (kNum + kBatchSize - 1) / kBatchSize;

    for (int32_t batch = 0; batch < num_batches; batch++) {
        int32_t batch_start = batch * kBatchSize;
        int32_t batch_count = (batch_start + kBatchSize <= kNum) ? kBatchSize : (kNum - batch_start);

        // For the last batch that may be smaller, we still launch with full tile
        // but only the valid elements matter
        LaunchHashFind<16, 16, kCapacity, kMaxProbe>(
            g_output + batch_start, g_hashtable_ro, g_query_keys + batch_start);
    }

    // Verify results
    int match = 0;
    int mismatch_count = 0;
    for (int i = 0; i < kNum; i++) {
        if (g_output[i] == g_lookup_values[i]) {
            match++;
        } else {
            mismatch_count++;
        }
    }

#ifndef FOR_GFSIM
    printf("=== hashfind tile-based lookup ===\n");
    printf("Match: %d/%d (%.4f%%)\n", match, kNum, 100.0 * double(match) / double(kNum));

    if (mismatch_count > 0) {
        printf("\n=== First 20 mismatches ===\n");
        printf("%7s  %22s  %10s  %10s\n", "Idx", "Key", "Got", "Expected");
        int shown = 0;
        for (int i = 0; i < kNum && shown < 20; i++) {
            if (g_output[i] != g_lookup_values[i]) {
                printf("%7d  %22lld  %10d  %10d\n",
                       i, (long long)g_query_keys[i], (int)g_output[i], (int)g_lookup_values[i]);
                shown++;
            }
        }
    }
    fflush(stdout);
#endif

    int ret = (match == kNum) ? 0 : 1;
#ifndef FOR_GFSIM
    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
#endif
    return ret;
}