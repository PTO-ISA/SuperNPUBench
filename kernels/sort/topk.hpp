#ifndef TOPK_HPP
#define TOPK_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// Constants
// ============================================================================

constexpr int kInputCount = 131072;
constexpr int kTopK       = 2048;
constexpr int kTileSize   = 256;
constexpr int kNumTiles   = kInputCount / kTileSize;
constexpr int kNumBuckets = 256;

// ============================================================================
// Tile type aliases
// ============================================================================

using TileU16 = Tile<Location::Vec, uint16_t, 1, kTileSize, BLayout::RowMajor>;
using TileU32 = Tile<Location::Vec, uint32_t, 1, kNumBuckets, BLayout::RowMajor>;

using InputGM = GlobalTensor<uint16_t, Shape<1,1,1,1,kTileSize>, Stride<1,1,1,kTileSize,1>>;
using HistGM  = GlobalTensor<uint32_t, Shape<1,1,1,1,kNumBuckets>, Stride<1,1,1,kNumBuckets,1>>;

// ============================================================================
// THISTOGRAM comes from the compiler's tileop-api (pulled in via pto_tileop.hpp):
//   void THISTOGRAM(dst, src, Idx, ByteId)
// This kernel depends on the upstream fix for the off-by-one operand numbering
// in tileop-api's THISTOGRAM template (template_asm.hpp) — tracked upstream in
// LinxISA/llvm-project. No self-contained copy is kept here on purpose; the
// tileop-api is owned by the compiler.
// ============================================================================

// ============================================================================
// Phase 1: Build high8 cumulative histogram via THISTOGRAM (Byte1, no filter)
// ============================================================================

inline void build_high8_histogram(uint16_t* input, uint32_t hist[256]) {
    TileU16 inputTile;
    TileU16 dummyIdx;
    TileU32 histTile, accumTile;

    TEXPANDSCALAR(accumTile, (uint32_t)0);
    TEXPANDSCALAR(dummyIdx, (uint16_t)0);

    for (int t = 0; t < kNumTiles; t++) {
        InputGM gm(input + t * kTileSize);
        TCOPYIN(inputTile, gm);
        THISTOGRAM(histTile, inputTile, dummyIdx, 1);
        TADD(accumTile, accumTile, histTile);
    }

    HistGM histGM(hist);
    TCOPYOUT(histGM, accumTile);
}

// ============================================================================
// Phase 3: Build low8 cumulative histogram for kth_bin via THISTOGRAM (Byte0, filtered)
// ============================================================================

inline void build_low8_histogram(uint16_t* input, uint16_t kth_bin, uint32_t hist[256]) {
    TileU16 inputTile, idxTile;
    TileU32 histTile, accumTile;

    TEXPANDSCALAR(accumTile, (uint32_t)0);
    TEXPANDSCALAR(idxTile, kth_bin);

    for (int t = 0; t < kNumTiles; t++) {
        InputGM gm(input + t * kTileSize);
        TCOPYIN(inputTile, gm);
        THISTOGRAM(histTile, inputTile, idxTile, 0);
        TADD(accumTile, accumTile, histTile);
    }

    HistGM histGM(hist);
    TCOPYOUT(histGM, accumTile);
}

// ============================================================================
// Phase 5: Masked select (scalar — TCMP doesn't support u16, TCAST crashes
// the compiler's ClockHands pass. The histogram phases above are the
// compute-intensive part and use THISTOGRAM tileblock ops.)
// ============================================================================

inline void masked_select(const uint16_t* input, int kth_bin, int low8_boundary, uint16_t* output) {
    for (int i = 0; i < kInputCount; i++) {
        uint16_t val  = input[i];
        uint8_t  high8 = static_cast<uint8_t>(val >> 8);
        int      low8  = static_cast<int>(val & 0xFF);
        int include = (high8 > kth_bin) ||
                      ((high8 == kth_bin) & (low8 >= low8_boundary));
        output[i] = include ? val : 0;
    }
}

// ============================================================================
// Scalar: find kth_bin from cumulative histogram
// C[k] = count of elements with byte value 0..k (output of THISTOGRAM)
// Scans from bin 255 down to find where cumulative-from-top first reaches k
// ============================================================================

static void find_kth_bin_from_cumulative(const uint32_t C[256], int k,
                                          int& kth_bin, int& need_from_kth) {
    uint32_t total = C[255];
    for (int b = 255; b >= 0; b--) {
        uint32_t count_ge = (b > 0) ? (total - C[b-1]) : total;
        if (count_ge >= (uint32_t)k) {
            kth_bin = b;
            need_from_kth = k - (int)(total - C[b]);
            return;
        }
    }
    kth_bin = 0;
    need_from_kth = k;
}

#endif // TOPK_HPP
