#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include <linxStartEnd.hpp>
#endif

#ifndef ROW
#define ROW 16
#endif

#ifndef COL
#define COL 16
#endif

#ifndef TROW
#define TROW 8
#endif

#ifndef TCOL
#define TCOL 8
#endif

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void store_nd2nd(float *dst) {
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;
    using gm_shape   = global_tensor<float, RowMajor<gm_row, gm_col>>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
        tile_shape d0(i+j);
        auto dstO = gdst(i,j);
        TSTORE(dstO, d0);
        }
    }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void store_nz2nd(float *dst) {
    using tile_shape = TileLeft<float, tile_row, tile_col>;
    using gm_shape   = global_tensor<float, RowMajor<gm_row, gm_col>>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gdst(dst);
    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
        tile_shape d0(i+j);
        auto dstO = gdst(i,j);
        TSTORE(dstO, d0);
        }
    }
}

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void store_zn2dn(float *dst) {
    // using tile_shape = TileRight<float, tile_row, tile_col>;
    // using gm_shape   = global_tensor<float, ColMajor<gm_row, gm_col>>;

    // uint16_t block_row = gm_row / tile_row;
    // uint16_t block_col = gm_col / tile_col;

    // using iter = global_iterator<gm_shape, tile_shape>;

    // iter gdst(dst);
    // for (int i = 0; i < block_row; ++i) {
    //     for (int j = 0; j < block_col; ++j) {
    //     tile_shape d0(i+j);
    //     auto dstO = gdst(i,j);
    //     TSTORE(dstO, d0);
    //     }
    // }
}

int main() {
    const uint16_t gm_row = ROW;
    const uint16_t gm_col = COL;
    const uint16_t tile_row = TROW;
    const uint16_t tile_col = TCOL;

    #ifdef LINX_PMC
    PMC_START();
    #endif

    size_t gm_size = gm_row * gm_col;

    float dst[gm_size];

    if(!strcmp(MODE, "ND2ND")){
        store_nd2nd<gm_row, gm_col, tile_row, tile_col>(dst);
    }else if(!strcmp(MODE, "NZ2ND")){
        store_nz2nd<gm_row, gm_col, tile_row, tile_col>(dst);
    }else if(!strcmp(MODE, "ZN2DN")){
        store_zn2dn<gm_row, gm_col, tile_row, tile_col>(dst);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}