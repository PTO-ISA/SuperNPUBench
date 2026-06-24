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
void tfillpad_nd(T *dst, T *src) {
    using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
    using tile_shape_in = Tile<Location::Vec, T, gm_row, gm_col, BLayout::RowMajor, tile_row, tile_col>;
    using tile_shape_out = Tile<Location::Vec, T, gm_row, gm_col, BLayout::RowMajor, gm_row, gm_col,
                                SLayout::NoneBox, 512, PadValue::Zero>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape_in>;

    iter gsrc(src);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in td0(2*i+j);
            tile_shape_out td1;
            TFILLPAD(td1, td0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void tfillpad_dn(T *dst, T *src) {
    using gm_shape = global_tensor<T, ColMajor<gm_row, gm_col>>;
    using tile_shape_in = Tile<Location::Vec, T, gm_row, gm_col, BLayout::ColMajor, tile_row, tile_col>;
    using tile_shape_out = Tile<Location::Vec, T, gm_row, gm_col, BLayout::ColMajor, gm_row, gm_col,
                                SLayout::NoneBox, 512, PadValue::Zero>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape_in>;

    iter gsrc(src);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in td0(2*i+j);
            tile_shape_out td1;
            TFILLPAD(td1, td0);
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
        tfillpad_nd<gm_row, gm_col, tile_row, tile_col>(dst, src);
    }else if(!strcmp(MODE, "DN")){
        tfillpad_dn<gm_row, gm_col, tile_row, tile_col>(dst, src);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}