#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// =====================================================================
// Broadcast (B,1,K) -> (B,N,K) via TLOAD + __vec__ broadcast + TSTORE
//
// Optimized for: (8192,1,16) -> (8192,8,16), dtype=half
//
// Data layout (row-major):
//   Input:  [B][1][K] -> flat = B*K,       batch b data at offset b*K
//   Output: [B][N][K] -> flat = B*N*K,     batch b data at offset b*N*K
//   Broadcast along dim1: 1 -> N, dim0 & dim2 preserved.
//
// Processing strategy:
//   Divide B batches into tiles of kTileBatch batches each.
//   Per tile:
//     1. TLOAD  (kTileBatch, K)   from GlobalMem -> TileReg
//        Reads kTileBatch * K contiguous elements.
//     2. __vec__ broadcast within TileReg:
//        For each batch, replicate its K elements N times (row-wise).
//        Launch <<<N*K, kTileBatch, 1>>> threads:
//          x = output position within one batch (0..N*K-1)
//          y = batch index within tile          (0..kTileBatch-1)
//          copy = x / K (broadcast copy 0..N-1)
//          col  = x % K (inner column   0..K-1)
//          Read  src[y * RowStride + col]
//          Write dst[y * RowStride + x]
//     3. TSTORE (kTileBatch, N*K)  from TileReg -> GlobalMem
//
// TileReg layout:
//   Physical tile cols = 256 (padded for 512B alignment).
//   Tile bytes = kTileBatch * 256 * sizeof(dtype) = 512 * kTileBatch
//   Constraint: kTileBatch must be power-of-2.
//
// Alignment notes:
//   K and N must be powers of 2 for bit-op index decomposition.
//   For 512B-aligned global tile strides (dtype=half):
//     Input stride  = kTileBatch * K * sizeof(half) = kTileBatch * K * 2
//                    512-aligned when kTileBatch * K * 2 >= 512 and is 512 * 2^n
//                    e.g. kTileBatch=16, K=16 -> 512B per tile  (512-aligned)
//     Output stride = kTileBatch * N * K * sizeof(half)
//                    e.g. kTileBatch=16, N=8, K=16 -> 4096B per tile (512-aligned)
//   For (8192,1,16)->(8192,8,16) with kTileBatch=16:
//     8192/16 = 512 tiles, no remainder, all aligned.
//
// Template params:
//   dtype      - data type (__half, float, etc.)
//   MAX_DIM    - max dimensions (kept for compat)
//   IN_DIM     - input dim count (kept for compat)
//   OUT_DIM    - output dim count (kept for compat)
//   gIM        - total input elements  = B * K      (e.g. 8192*16 = 131072)
//   gOM        - total output elements = B * N * K  (e.g. 8192*8*16 = 1048576)
//   kTileBatch - batches per tile, power-of-2 (e.g. 1,2,4,8,16,32,64)
//   kInner     - inner dimension K, power-of-2      (e.g. 16)
// =====================================================================

template <typename tile_shape_out, typename tile_shape_in, size_t kInnerCols>
void __vec__
vec_broadcast_3d(typename tile_shape_out::TileDType __out__ dst,
                 const typename tile_shape_in::TileDType __in__ src) {
    size_t x = blkv_get_index_x();
    size_t y = blkv_get_index_y();

    size_t in_index  = y * tile_shape_in::RowStride + (x & (kInnerCols - 1));
    size_t out_index = y * tile_shape_out::RowStride + x;

    blkv_get_tile_ptr(dst)[out_index] = blkv_get_tile_ptr(src)[in_index];
}

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileBatch, size_t kInner>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kBCast = gOM / gIM; // 8
    constexpr size_t kBCast_in = 64 / kInner; // 2
    constexpr size_t kBCast_out = kBCast / kBCast_in; // 4
    constexpr size_t kBatch = gIM / kInner;
    constexpr size_t tileCols = 256;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (B,1,K)->(B,N,K) broadcast");
    static_assert(gIM % kInner == 0,
                  "gIM must be divisible by kInner (B = gIM/kInner must be integer)");
    static_assert((kInner & (kInner - 1)) == 0,
                  "kInner must be power of 2 for bit-op index decomposition");
    static_assert((kTileBatch & (kTileBatch - 1)) == 0,
                  "kTileBatch must be power of 2 for 512B tile alignment");
    static_assert(tileCols >= kBCast * kInner,
                  "padded tileCols (256) must >= broadcast target width (N*K)");

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
        TLOAD(inTile, gsrc);

        vec_broadcast_3d<tile_out, tile_in, kInner>
            <<<kBCast * kInner, kTileBatch, 1>>>(outTile.data(), inTile.data());

        gm_out gdst(out_ptr + i * kTileBatch * kBCast * kInner);
        TSTORE(gdst, outTile);
    }

    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileBatch * kInner);
        TLOAD(inTile_rmd, gsrc);

        vec_broadcast_3d<tile_out_r, tile_in_r, kInner>
            <<<kBCast * kInner, rmd, 1>>>(outTile_rmd.data(), inTile_rmd.data());

        gm_out gdst(out_ptr + Nb * kTileBatch * kBCast * kInner);
        TSTORE(gdst, outTile_rmd);
    }
}