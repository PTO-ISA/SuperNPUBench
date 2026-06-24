#include <common/pto_tileop.hpp>
#include <string>

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

#ifndef MODE
#define MODE "ND"
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void texpandscalar_nd(float *dst, float s) {
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gdst(i, j);

            tile_shape td0;
            TEXPANDSCALAR(td0, s);
            // TSTORE(g0, td0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void texpandscalar_dn(float *dst, float s) {
    using gm_shape = global_tensor<float, ColMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::ColMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gdst(i, j);

            tile_shape td0;
            TEXPANDSCALAR(td0, s);
            // TSTORE(g0, td0);
        }
    }
}

int main() {
    const uint16_t gm_row = ROW;
    const uint16_t gm_col = COL;
    const uint16_t tile_row = TROW;
    const uint16_t tile_col = TCOL;

    size_t gm_size = gm_row * gm_col;
    size_t tile_size = tile_row * tile_col;

    float s_fp32 = 0.1;
    float dst[gm_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        texpandscalar_nd<gm_row, gm_col, tile_row, tile_col>(dst, s_fp32);
    }else if(!strcmp(MODE, "DN")){
        texpandscalar_dn<gm_row, gm_col, tile_row, tile_col>(dst, s_fp32);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}