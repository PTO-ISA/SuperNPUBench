#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

// =====================================================================
// Broadcast (N,1) -> (N,C) via TCOPYIN + __vec__ broadcast + TCOPYOUT
//
// Optimized for: (1443,1) -> (1443,129), dtype=half
//
// Processing strategy:
//   Divide N rows into tiles of kTileRows rows each.
//   Per tile:
//     1. TCOPYIN   (kTileRows, 1)   from GlobalMem -> TileReg
//     2. __vec__ broadcast (kTileRows, 1) -> (kTileRows, C) within TileReg
//        Launch <<<kC, kTileRows, 1>>> threads:
//          x = column index (0..kC-1), y = row index (0..kTileRows-1)
//          Each thread reads src[j*src_RowStride] (col 0 of row j)
//          and writes to dst[i + j*dst_RowStride] (col i of row j)
//     3. TCOPYOUT  (kTileRows, C)   from TileReg  -> GlobalMem
//
// TileReg layout:
//   Physical tile cols padded to 256 for 512B alignment.
//   Tile bytes = kTileRows * 256 * sizeof(dtype) = 512 * kTileRows
//   Constraint: kTileRows must be power-of-2  (512 * 2^n bytes).
//
// Template params (backward compatible with original broadcast_07):
//   dtype     - data type (__half, float, etc.)
//   MAX_DIM   - max dimensions (unused, kept for compat)
//   IN_DIM    - input dim count (unused)
//   OUT_DIM   - output dim count (unused)
//   gIM       - total input elements = N*1 = N   (e.g. 1443)
//   gOM       - total output elements = N*C       (e.g. 1443*129)
//   kTileRows - rows per tile, must be power-of-2 (e.g. 1,2,4,..,64)
// =====================================================================

template <typename tile_shape_out, typename tile_shape_in>
void __vec__
vec_broadcast_rowmajor(typename tile_shape_out::TileDType __out__ dst,
                       const typename tile_shape_in::TileDType __in__ src) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t out_index = i + j * tile_shape_out::RowStride;
    blkv_get_tile_ptr(dst)[out_index] =
        blkv_get_tile_ptr(src)[j * tile_shape_in::RowStride];
}

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileRows>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kN = gIM;
    constexpr size_t kC = gOM / gIM;
    constexpr size_t tileCols = 256;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (N,1)->(N,C) broadcast");
    static_assert(tileCols >= kC,
                  "padded tileCols (256) must >= broadcast target columns");
    static_assert((kTileRows & (kTileRows - 1)) == 0,
                  "kTileRows must be power of 2 for 512B tile alignment");

    using tile_in  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, 1>;
    using tile_out = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, kC>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileRows, 1>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileRows, kC>>;

    constexpr size_t Nb  = kN / kTileRows;
    constexpr size_t rmd = kN % kTileRows;

    tile_in inTile;
    tile_out outTile;


    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileRows);
        TCOPYIN(inTile, gsrc);

        vec_broadcast_rowmajor<tile_out, tile_in>
            <<<kC, kTileRows, 1>>>(outTile.data(), inTile.data());

        gm_out gdst(out_ptr + i * kTileRows * kC);
        TCOPYOUT(gdst, outTile);
    }

    using tile_in_r  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, 1>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, kC>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;
    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileRows);
        TCOPYIN(inTile_rmd, gsrc);

        vec_broadcast_rowmajor<tile_out_r, tile_in_r>
            <<<kC, rmd, 1>>>(outTile_rmd.data(), inTile_rmd.data());

        gm_out gdst(out_ptr + Nb * kTileRows * kC);
        TCOPYOUT(gdst, outTile_rmd);
    }
}