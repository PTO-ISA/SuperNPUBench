#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "template_asm.h"
#include <stdio.h>

// ============================================================================
// Tile operation implementations
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
// Constants
// ============================================================================

constexpr uint32_t kCapacity = 2048;
constexpr uint32_t kC1 = 0xcc9e2d51;
constexpr uint32_t kC2 = 0x1b873593;
constexpr uint32_t kC3 = 0xe6546b64;
constexpr uint32_t kFinal1 = 0x85ebca6b;
constexpr uint32_t kFinal2 = 0xc2b2ae35;
constexpr int64_t kEmptyKey = 0x8000000000000000LL;
constexpr int32_t kNotFound = -1;

// ============================================================================
// Data layout
// ============================================================================

// 16 bytes per slot: key(int64) + value(int32) + padding(int32)
struct TableEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

extern "C" {
    extern const uint8_t _binary_simple_inserted_slot_data_start[];
    extern const uint8_t _binary_simple_inserted_slot_data_end[];
    extern const uint8_t _binary_simple_lookup_keys_data_start[];
    extern const uint8_t _binary_simple_lookup_keys_data_end[];
    extern const uint8_t _binary_simple_lookup_values_data_start[];
    extern const uint8_t _binary_simple_lookup_values_data_end[];
}

static TableEntry* g_hashtable_ro = reinterpret_cast<TableEntry*>(
    const_cast<uint8_t*>(_binary_simple_inserted_slot_data_start));
// Writable copy of the hash table — the .data.rel.ro section is not writable
static TableEntry g_hashtable[kCapacity];
static int64_t* g_query_keys = reinterpret_cast<int64_t*>(
    const_cast<uint8_t*>(_binary_simple_lookup_keys_data_start));
static int32_t* g_lookup_values = reinterpret_cast<int32_t*>(
    const_cast<uint8_t*>(_binary_simple_lookup_values_data_start));

static int32_t g_output[256];

// 256 distinct update values (one per lane) — used as MSCATTER source
static int32_t g_update_values[256] = {
    101,    200,    302,    413,    5,    6,    7,    8,    9,   10,
   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,
   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,
   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,
   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,
   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,
   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,
   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,
   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,
   91,   92,   93,   94,   95,   96,   97,   98,   99,  100,
  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,
  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,
  121,  122,  123,  124,  125,  126,  127,  128,  129,  130,
  131,  132,  133,  134,  135,  136,  137,  138,  139,  140,
  141,  142,  143,  144,  145,  146,  147,  148,  149,  150,
  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,
  161,  162,  163,  164,  165,  166,  167,  168,  169,  170,
  171,  172,  173,  174,  175,  176,  177,  178,  179,  180,
  181,  182,  183,  184,  185,  186,  187,  188,  189,  190,
  191,  192,  193,  194,  195,  196,  197,  198,  199,  200,
  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,
  211,  212,  213,  214,  215,  216,  217,  218,  219,  220,
  221,  222,  223,  224,  225,  226,  227,  228,  229,  230,
  231,  232,  233,  234,  235,  236,  237,  238,  239,  240,
  241,  242,  243,  244,  245,  246,  247,  248,  249,  250,
  251,  252,  253,  254,  255,  256
};

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

// MGATHER offset arrays for extracting low/high 32-bit halves from int64_t keys
// Each query key is 8 bytes: low at offsets 0,8,16,... and high at 4,12,20,...
static uint16_t g_offset_low[256];
static uint16_t g_offset_high[256];
static bool g_offsets_init = false;

static void init_offset_arrays() {
    if (!g_offsets_init) {
        for (int i = 0; i < 256; i++) {
            g_offset_low[i]  = static_cast<uint16_t>(i * 8);
            g_offset_high[i] = static_cast<uint16_t>(i * 8 + 4);
        }
        g_offsets_init = true;
    }
}

// ============================================================================
// Step 1: Load query keys (extract low/high 32-bit halves + full 64-bit key)
// ============================================================================

template <int kTileRows, int kTileCols, int kCap>
void loadKeys(typename HashFindTypes<kTileRows, kTileCols>::TileU32& lowTile,
              typename HashFindTypes<kTileRows, kTileCols>::TileU32& highTile,
              typename HashFindTypes<kTileRows, kTileCols>::TileI64& queryKeyTile,
              int64_t __in__ *queries)
{
    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileU16 = typename Types::TileU16;
    using OffsetShape  = Shape<1, 1, 1, kTileRows, kTileCols>;
    using OffsetStride = Stride<1, 1, 1, kTileCols, 1>;
    using OffsetGT = GlobalTensor<uint16_t, OffsetShape, OffsetStride>;
    using KeyShape  = Shape<1, 1, 1, kTileRows, kTileCols>;
    using KeyStride = Stride<1, 1, 1, kTileCols, 1>;
    using KeyGT = GlobalTensor<int64_t, KeyShape, KeyStride>;

    init_offset_arrays();

    TileU16 offsetLowTile, offsetHighTile;
    OffsetGT offsetLowGlobal(g_offset_low);
    OffsetGT offsetHighGlobal(g_offset_high);
    TLOAD(offsetLowTile,  offsetLowGlobal);
    TLOAD(offsetHighTile, offsetHighGlobal);

    KeyGT keysGlobal(queries);
    MGATHER(lowTile,  keysGlobal, offsetLowTile);
    MGATHER(highTile, keysGlobal, offsetHighTile);

    TLOAD(queryKeyTile, keysGlobal);
}

// ============================================================================
// Step 2: Compute MurmurHash3 and initial probe index
// ============================================================================

template <int kTileRows, int kTileCols, int kCap>
void computeHash(typename HashFindTypes<kTileRows, kTileCols>::TileU32& hTile,
                 typename HashFindTypes<kTileRows, kTileCols>::TileU32& probeIdxTile,
                 typename HashFindTypes<kTileRows, kTileCols>::TileU32& lowTile,
                 typename HashFindTypes<kTileRows, kTileCols>::TileU32& highTile)
{
    using TileU32 = typename HashFindTypes<kTileRows, kTileCols>::TileU32;

    TileU32 shift15, shift17, shift13, shift19, shift16;
    TileU32 tmpU32, k1Tile, k1RotL, hRotL;
    TileU32 eightTile;

    TEXPANDSCALAR(shift15, static_cast<uint32_t>(15));
    TEXPANDSCALAR(shift17, static_cast<uint32_t>(17));
    TEXPANDSCALAR(shift13, static_cast<uint32_t>(13));
    TEXPANDSCALAR(shift19, static_cast<uint32_t>(19));
    TEXPANDSCALAR(shift16, static_cast<uint32_t>(16));
    TEXPANDSCALAR(eightTile, static_cast<uint32_t>(8));
    TEXPANDSCALAR(hTile,   static_cast<uint32_t>(0));

    // Block 1: low 32 bits
    k1Tile = lowTile;
    TMULS(k1Tile, k1Tile, kC1);
    TSHL_Impl(k1RotL, k1Tile, shift15);
    TSHR_Impl(tmpU32,  k1Tile, shift17);
    TOR_Impl(k1Tile, k1RotL, tmpU32);
    TMULS(k1Tile, k1Tile, kC2);
    TXOR_Impl(hTile, hTile, k1Tile);
    TSHL_Impl(hRotL, hTile, shift13);
    TSHR_Impl(tmpU32,  hTile, shift19);
    TOR_Impl(hTile, hRotL, tmpU32);
    TMULS(hTile, hTile, static_cast<uint32_t>(5));
    TADDS(hTile, hTile, kC3);

    // Block 2: high 32 bits
    k1Tile = highTile;
    TMULS(k1Tile, k1Tile, kC1);
    TSHL_Impl(k1RotL, k1Tile, shift15);
    TSHR_Impl(tmpU32,  k1Tile, shift17);
    TOR_Impl(k1Tile, k1RotL, tmpU32);
    TMULS(k1Tile, k1Tile, kC2);
    TXOR_Impl(hTile, hTile, k1Tile);
    TSHL_Impl(hRotL, hTile, shift13);
    TSHR_Impl(tmpU32,  hTile, shift19);
    TOR_Impl(hTile, hRotL, tmpU32);
    TMULS(hTile, hTile, static_cast<uint32_t>(5));
    TADDS(hTile, hTile, kC3);

    // Finalization
    TXOR_Impl(hTile, hTile, eightTile);
    TSHR_Impl(tmpU32, hTile, shift16);
    TXOR_Impl(hTile, hTile, tmpU32);
    TMULS(hTile, hTile, kFinal1);
    TSHR_Impl(tmpU32, hTile, shift13);
    TXOR_Impl(hTile, hTile, tmpU32);
    TMULS(hTile, hTile, kFinal2);
    TSHR_Impl(tmpU32, hTile, shift16);
    TXOR_Impl(hTile, hTile, tmpU32);

    // Initial probe index: hash & (capacity - 1)
    TileU32 capMinus1;
    TEXPANDSCALAR(capMinus1, static_cast<uint32_t>(kCap - 1));
    TAND(probeIdxTile, hTile, capMinus1);
}

// ============================================================================
// Step 3: Linear probing (kMaxProbe iterations)
// ============================================================================

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void linearProbing(typename HashFindTypes<kTileRows, kTileCols>::TileI32& outTile,
                   typename HashFindTypes<kTileRows, kTileCols>::TileU32& probeIdxTile,
                   typename HashFindTypes<kTileRows, kTileCols>::TileI64& queryKeyTile,
                   TableEntry __in__ *table,
                   typename HashFindTypes<kTileRows, kTileCols>::TileU32& foundIdxTile)
{
    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileU16 = typename Types::TileU16;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;
    using TableKeyGT = GlobalTensor<int64_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;
    // tableValGlobal base is already table + 8 (pointing to value field of entry[0])
    using TableValGT = GlobalTensor<int32_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;

    TableKeyGT tableKeyGlobal(reinterpret_cast<int64_t*>(table));
    TableValGT tableValGlobal(reinterpret_cast<int32_t*>(reinterpret_cast<char*>(table) + 8));

    TileU32 capMinus1, sixteenTile;
    TEXPANDSCALAR(capMinus1,  static_cast<uint32_t>(kCap - 1));
    TEXPANDSCALAR(sixteenTile, static_cast<uint32_t>(16));

    TileU32 oneTile;
    TEXPANDSCALAR(oneTile, static_cast<uint32_t>(1));

    // Initialize foundIdxTile with kCap sentinel (out of range for a 2048-entry table)
    TEXPANDSCALAR(foundIdxTile, static_cast<uint32_t>(kCap));

    TileU32 byteOffsetTile;
    TileU16 byteOffsetTile16;
    TileI64 tableKeyTile;
    TileI32 tableValueTile;
    TileI32 probeMatchi32;
    TileI64 probeMatchi64;

    for (int probe = 0; probe < kMaxProbe; probe++) {
        TMUL(byteOffsetTile, probeIdxTile, sixteenTile);
        TCAST(byteOffsetTile16, byteOffsetTile);

        // Load table keys
        MGATHER(tableKeyTile, tableKeyGlobal, byteOffsetTile16);

        // tableValGlobal base = table + 8, so byteOffsetTile indexes values directly
        MGATHER(tableValueTile, tableValGlobal, byteOffsetTile16);

        // Select value where key matches; keep previous result otherwise
        TCMP_Impl(probeMatchi64, tableKeyTile, queryKeyTile);
        TCAST(probeMatchi32, probeMatchi64);
        TSELECT(outTile, probeMatchi32, tableValueTile, outTile);

        // Record the index where this key was found (used for MSCATTER update)
        TSELECT(foundIdxTile, probeMatchi32, probeIdxTile, foundIdxTile);

        // Advance probe index (wrap around capacity)
        TADD(probeIdxTile, probeIdxTile, oneTile);
        TAND(probeIdxTile, probeIdxTile, capMinus1);
    }
}

// ============================================================================
// Step 4: Update table values via MSCATTER (write back found values)
// ============================================================================

template <int kTileRows, int kTileCols, int kCap>
void updateValues(typename HashFindTypes<kTileRows, kTileCols>::TileI32& updateTile,
                  typename HashFindTypes<kTileRows, kTileCols>::TileU32& foundIdxTile,
                  TableEntry __in__ *table)
{
    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileU16 = typename Types::TileU16;
    using TileU32 = typename Types::TileU32;
    // tableValGlobal base is table + 8 (pointing to value field of entry[0])
    using TableValGT = GlobalTensor<int32_t, Shape<1,1,1,1,kCap>, Stride<1,1,1,1,1>>;

    TableValGT tableValGlobal(reinterpret_cast<int32_t*>(reinterpret_cast<char*>(table) + 8));

    TileU32 byteOffsetTile;
    TileU16 byteOffsetTile16;
    TMULS(byteOffsetTile, foundIdxTile, static_cast<uint32_t>(16));
    TCAST(byteOffsetTile16, byteOffsetTile);
    MSCATTER(tableValGlobal, updateTile, byteOffsetTile16);
}

// ============================================================================
// runHashFind - processes one batch of 256 keys
// ============================================================================

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void runHashFind(int32_t __out__ *out,
                 TableEntry __in__ *table,
                 int64_t __in__ *queries,
                 int32_t __in__ *expected_values,
                 bool do_update,
                 typename HashFindTypes<kTileRows, kTileCols>::TileI32& updateTile,
                 int32_t __in__ *update_values)
{
    (void)expected_values;
    static_assert((kCap & (kCap - 1)) == 0, "hashfind: capacity must be power-of-two");

    using Types = HashFindTypes<kTileRows, kTileCols>;
    using TileU32 = typename Types::TileU32;
    using TileU16 = typename Types::TileU16;
    using TileI64 = typename Types::TileI64;
    using TileI32 = typename Types::TileI32;
    using TileGT = GlobalTensor<int32_t, Shape<1,1,1,kTileRows,kTileCols>, Stride<1,1,1,kTileCols,1>>;

    TileU32 lowTile, highTile, hTile, probeIdxTile, foundIdxTile;
    TileI64 queryKeyTile;
    TileI32 outTile;
    TEXPANDSCALAR(outTile, kNotFound);

    loadKeys<kTileRows, kTileCols, kCap>(lowTile, highTile, queryKeyTile, queries);
    computeHash<kTileRows, kTileCols, kCap>(hTile, probeIdxTile, lowTile, highTile);

    linearProbing<kTileRows, kTileCols, kCap, kMaxProbe>(
        outTile, probeIdxTile, queryKeyTile, table, foundIdxTile);

    if (do_update) {
        // Load the 256 distinct update values, then MSCATTER to the table
        using UpdGT = GlobalTensor<int32_t, Shape<1,1,1,kTileRows,kTileCols>, Stride<1,1,1,kTileCols,1>>;
        UpdGT updGlobal(update_values);
        TLOAD(updateTile, updGlobal);
        updateValues<kTileRows, kTileCols, kCap>(updateTile, foundIdxTile, table);
    }

    TileGT outGlobal(out);
    TSTORE(outGlobal, outTile);
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t *out, TableEntry* table, int64_t* queries,
                    int32_t* expected_values, void *stream, bool do_update,
                    typename HashFindTypes<kTileRows, kTileCols>::TileI32& updateTile,
                    int32_t* update_values)
{
    (void)stream;
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(
        out, table, queries, expected_values, do_update, updateTile, update_values);
}

// ============================================================================
// main
// ============================================================================

int main() {
    // Copy read-only table data into our writable buffer so MSCATTER can modify it
    for (uint32_t i = 0; i < kCapacity; i++) {
        g_hashtable[i] = g_hashtable_ro[i];
    }

    // Save a copy of the original values before any MSCATTER updates
    int32_t g_original_values[256];
    for (int i = 0; i < 256; i++) {
        g_original_values[i] = g_lookup_values[i];
    }

    // Tile for MSCATTER source data (only used when do_update=true, but
    // declared here so all three kernel calls have a consistent signature)
    HashFindTypes<16, 16>::TileI32 updateTile;

    // --- Round 1: initial lookup (no MSCATTER update) ---
    LaunchHashFind<16, 16, kCapacity, 8>(
        g_output, g_hashtable, g_query_keys, g_lookup_values,
        nullptr, false, updateTile, g_update_values);

    int match_before = 0;
    for (int i = 0; i < 256; i++) {
        int32_t expected = g_original_values[i];
        int32_t kernel   = g_output[i];
        if (kernel == expected) match_before++;
    }

    printf("=== Round 1: Initial Lookup (before MSCATTER update) ===\n");
    printf("Kernel output:    %d/256 (%.1f%%)\n", match_before, 100.0 * match_before / 256);
    printf("  (first 5 results: kernel=[");
    for (int i = 0; i < 5; i++) {
        printf("%d%s", g_output[i], i < 4 ? ", " : "");
    }
    printf("])\n");
    fflush(stdout);

    // --- Round 2: MSCATTER update (write g_update_values to the table) ---
    LaunchHashFind<16, 16, kCapacity, 8>(
        g_output, g_hashtable, g_query_keys, g_lookup_values,
        nullptr, true, updateTile, g_update_values);

    // --- Round 3: lookup again to verify MSCATTER wrote g_update_values ---
    LaunchHashFind<16, 16, kCapacity, 8>(
        g_output, g_hashtable, g_query_keys, g_lookup_values,
        nullptr, false, updateTile, g_update_values);

    // After MSCATTER, the table holds g_update_values, so re-lookup should
    // return those distinct values instead of the original ones
    int match_updated = 0;
    for (int i = 0; i < 256; i++) {
        int32_t expected_update = g_update_values[i];
        int32_t kernel   = g_output[i];
        if (kernel == expected_update) match_updated++;
    }

    printf("\n=== Round 3: Lookup After MSCATTER Update ===\n");
    printf("Kernel output:    %d/256 (%.1f%%)\n", match_updated, 100.0 * match_updated / 256);
    printf("  (first 5 results: kernel=[");
    for (int i = 0; i < 5; i++) {
        printf("%d%s", g_output[i], i < 4 ? ", " : "");
    }
    printf("])\n");
    printf("  (expected update values:[");
    for (int i = 0; i < 5; i++) {
        printf("%d%s", g_update_values[i], i < 4 ? ", " : "");
    }
    printf("])\n");
    fflush(stdout);

    int ret = (match_before == 256 && match_updated == 256) ? 0 : 1;
    if (!ret) {
        printf("PASS\n");
    }  else {
        printf("FAIL\n");
    }
    return ret;
}