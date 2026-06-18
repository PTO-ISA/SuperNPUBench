#include <common/pto_tileop.hpp>
#include <string> 

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
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

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void ttrans_nd(T *dst, T *src) {
    using gm_shape_in = global_tensor<T, RowMajor<gm_row, gm_col>>;
    using gm_shape_out = global_tensor<T, RowMajor<gm_col, gm_row>>;

    using tile_shape_in = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;
    using tile_shape_out = Tile<Location::Vec, T, tile_col, tile_row, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_in = global_iterator<gm_shape_in, tile_shape_in>;
    using iter_out = global_iterator<gm_shape_out, tile_shape_out>;

    iter_in gsrc(src);
    iter_out gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in td0(2*i+j);
            tile_shape_out td1;
            TTRANS(td1, td0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void ttrans_dn(T *dst, T *src) {
    using gm_shape_in = global_tensor<T, ColMajor<gm_row, gm_col>>;
    using gm_shape_out = global_tensor<T, ColMajor<gm_col, gm_row>>;

    using tile_shape_in = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor>;
    using tile_shape_out = Tile<Location::Vec, T, tile_col, tile_row, BLayout::ColMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_in = global_iterator<gm_shape_in, tile_shape_in>;
    using iter_out = global_iterator<gm_shape_out, tile_shape_out>;

    iter_in gsrc(src);
    iter_out gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in td0(2*i+j);
            tile_shape_out td1;
            TTRANS(td1, td0);
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

    float src[gm_size];
    float dst[gm_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        ttrans_nd<gm_row, gm_col, tile_row, tile_col>(dst, src);
    }else if(!strcmp(MODE, "DN")){
        ttrans_dn<gm_row, gm_col, tile_row, tile_col>(dst, src);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}