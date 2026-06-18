#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// =====================================================================
// Broadcast (B,1,K) -> (B,N,K) via TCOPYIN + __vec__ broadcast + TCOPYOUT
//
// Optimized for: (1280,1,49) -> (1280,8,49), dtype=half
//
// Data layout (row-major):
//   Input:  [B][1][K] -> flat = B*K,       batch b data at offset b*K
//   Output: [B][N][K] -> flat = B*N*K,     batch b data at offset b*N*K
//   Broadcast along dim1: 1 -> N, dim0 & dim2 preserved.
//
// Processing strategy:
//   Divide B batches into tiles of kTileBatch batches each.
//   Per tile:
//     1. TCOPYIN  (kTileBatch, K)   from GlobalMem -> TileReg
//        Reads kTileBatch * K contiguous elements.
//     2. __vec__ broadcast within TileReg:
//        Launch <<<K, N*kTileBatch, 1>>> threads:
//          x = inner column index           (0..K-1,   direct use, no modulo)
//          y = broadcast_copy * kTileBatch + batch_idx (0..N*kTileBatch-1)
//          copy      = y / kTileBatch        (0..N-1,   power-of-2 safe)
//          batch_idx = y & (kTileBatch - 1)  (0..kTileBatch-1, bitwise)
//          Read  src[batch_idx * RowStride + x]
//          Write dst[batch_idx * RowStride + copy * K + x]
//     3. TCOPYOUT (kTileBatch, N*K)  from TileReg -> GlobalMem
//
// TileReg layout:
//   Physical tile cols = 512 (padded for 512B alignment).
//   Constraint: kTileBatch must be power-of-2, N must be power-of-2.
//
// Note: kInner (K=49) is NOT power-of-2, so we avoid modulo on K by
//       making x the direct column index (0..K-1). All division on
//       y uses power-of-2 kTileBatch (bitwise ops only).
//
// Alignment notes (dtype=half):
//   kTileBatch=16, Cols=512 -> 16*512*2 = 16384 = 512*2^5 (aligned)
//   For (1280,1,49)->(1280,8,49) with kTileBatch=16:
//     1280/16 = 80 tiles, no remainder.
//
// Template params:
//   dtype      - data type (__half, float, etc.)
//   MAX_DIM    - max dimensions (kept for compat)
//   IN_DIM     - input dim count (kept for compat)
//   OUT_DIM    - output dim count (kept for compat)
//   gIM        - total input elements  = B * K
//   gOM        - total output elements = B * N * K
//   kTileBatch - batches per tile, power-of-2
//   kInner     - inner dimension K (e.g. 49, need not be power-of-2)
// =====================================================================

template <typename tile_shape_out, typename tile_shape_in,
          size_t kInnerCols, size_t kBCast, size_t kTileBatch>
void __vec__
vec_broadcast_3d(typename tile_shape_out::TileDType __out__ dst,
                  const typename tile_shape_in::TileDType __in__ src) {
    size_t x = blkv_get_index_x();
    size_t y = blkv_get_index_y();

    size_t batch_idx = y & (kTileBatch - 1);
    size_t copy      = y >> 4;

    size_t in_index  = batch_idx * tile_shape_in::RowStride + x;
    size_t out_index = batch_idx * tile_shape_out::RowStride + copy * kInnerCols + x;

    blkv_get_tile_ptr(dst)[out_index] = blkv_get_tile_ptr(src)[in_index];
}

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileBatch, size_t kInner>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kBCast = gOM / gIM;
    constexpr size_t kBatch = gIM / kInner;
    constexpr size_t tileCols = 512;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (B,1,K)->(B,N,K) broadcast");
    static_assert(gIM % kInner == 0,
                  "gIM must be divisible by kInner (B = gIM/kInner must be integer)");
    static_assert((kTileBatch & (kTileBatch - 1)) == 0,
                  "kTileBatch must be power of 2 for 512B tile alignment");
    static_assert((kBCast & (kBCast - 1)) == 0,
                  "kBCast (N) must be power of 2 for bitwise division in SIMT");
    static_assert(tileCols >= kBCast * kInner,
                  "padded tileCols (512) must >= broadcast target width (N*K)");

    using tile_in  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kInner>;
    using tile_out = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kBCast * kInner>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileBatch, kInner>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileBatch, kBCast * kInner>>;

    constexpr size_t Nb  = kBatch / kTileBatch;
    constexpr size_t rmd = kBatch % kTileBatch;

    tile_in inTile;
    tile_out outTile;

    using tile_in_r  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kInner>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kBCast * kInner>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;

    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileBatch * kInner);
        TCOPYIN(inTile, gsrc);

        vec_broadcast_3d<tile_out, tile_in, kInner, kBCast, kTileBatch>
            <<<kInner, kBCast * kTileBatch, 1>>>(outTile.data(), inTile.data());

        gm_out gdst(out_ptr + i * kTileBatch * kBCast * kInner);
        TCOPYOUT(gdst, outTile);
    }

    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileBatch * kInner);
        TCOPYIN(inTile_rmd, gsrc);

        vec_broadcast_3d<tile_out_r, tile_in_r, kInner, kBCast, rmd>
            <<<kInner, kBCast * rmd, 1>>>(outTile_rmd.data(), inTile_rmd.data());

        gm_out gdst(out_ptr + Nb * kTileBatch * kBCast * kInner);
        TCOPYOUT(gdst, outTile_rmd);
    }
}