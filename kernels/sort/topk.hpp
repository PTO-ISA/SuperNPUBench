#ifndef TOPK_HPP
#define TOPK_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// Constants
// ============================================================================

constexpr int kInputCount = 131072;
constexpr int kTopK       = 2048;
constexpr int kTileSize   = 256;        // tile register size (16×16)
constexpr int kNumTiles   = kInputCount / kTileSize;
constexpr int kNumBuckets = 256;

// ============================================================================
// Tile type aliases
// ============================================================================

using TileU16 = Tile<Location::Vec, uint16_t, 16, 16, BLayout::RowMajor>;
using TileU32 = Tile<Location::Vec, uint32_t, 16, 16, BLayout::RowMajor>;

// ============================================================================
// Phase 1: SIMT Extract high8 histogram (per-bucket)
//
// Grid: <<<1, 256, 1>>>  (1 block, 256 lanes, one lane per bucket 0..255)
// Each lane (bucket = lane_id):
//   loops over ALL kInputCount elements from global memory
//   counts how many have high8 == bucket
//   writes count to dst[lane_id].
// ============================================================================

template <typename tile_shape_out>
void __vec__ ExtractHigh8Hist_Vec_RowMajor(
    typename tile_shape_out::TileDType __out__ dst,
    const uint16_t* __in__ src)
{
    size_t bucket = blkv_get_index_y();
    typename tile_shape_out::DType count = 0;

    for (unsigned int i = 0; i < kInputCount; i++) {
        uint16_t val   = src[i];
        uint8_t  high8 = static_cast<uint8_t>(val >> 8);
        if (high8 == bucket) {
            count += 1;
        }
    }

    blkv_get_tile_ptr(dst)[bucket] = count;
}

// ============================================================================
// Phase 3: SIMT Extract low8 histogram for kth_bin elements (per-bucket)
// ============================================================================

template <typename tile_shape_out>
void __vec__ ExtractLow8HistForKthBin_Vec_RowMajor(
    typename tile_shape_out::TileDType __out__ dst,
    const uint16_t* __in__ src,
    uint16_t kth_bin)
{
    size_t bucket = blkv_get_index_y();
    typename tile_shape_out::DType count = 0;

    for (unsigned int i = 0; i < kInputCount; i++) {
        uint16_t val   = src[i];
        uint8_t  high8 = static_cast<uint8_t>(val >> 8);
        if (high8 == kth_bin) {
            uint8_t low8 = static_cast<uint8_t>(val & 0xFF);
            if (low8 == bucket) {
                count += 1;
            }
        }
    }

    blkv_get_tile_ptr(dst)[bucket] = count;
}

// ============================================================================
// Wrapper launch helpers
// ============================================================================

template <typename tile_shape_out>
void ExtractHigh8Hist_Impl(tile_shape_out& dst, const uint16_t* src) {
    ExtractHigh8Hist_Vec_RowMajor<tile_shape_out>
        <<<1, 256, 1>>>(dst.data(), src);
}

template <typename tile_shape_out>
void ExtractLow8HistForKthBin_Impl(tile_shape_out& dst, const uint16_t* src,
                                   uint16_t kth_bin) {
    ExtractLow8HistForKthBin_Vec_RowMajor<tile_shape_out>
        <<<1, 256, 1>>>(dst.data(), src, kth_bin);
}

// ============================================================================
// Scalar helper: prefix scan to find kth_bin and remaining count
// ============================================================================

static int find_kth_bin(const uint32_t hist[256], int k, int& need_from_kth) {
    // Scan from highest bin down — we want the largest k elements.
    uint64_t cumsum = 0;
    for (int b = 255; b >= 0; b--) {
        cumsum += hist[b];
        if (cumsum >= static_cast<uint64_t>(k)) {
            need_from_kth = k - static_cast<int>(cumsum - hist[b]);
            return b;
        }
    }
    need_from_kth = 0;
    return 0;
}

#endif // TOPK_HPP
