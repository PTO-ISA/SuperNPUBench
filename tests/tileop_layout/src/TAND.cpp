#include <common/pto_tileop.hpp>
#include <string> 
#include "jcore/TAnd.hpp"

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

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col, typename T>
void tand_nd(T *dst, T *src0, T *src1) {
    using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc0(src0);
    iter gsrc1(src1);
    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(i, j);
            auto g1 = gsrc1(i, j);
            auto g2 = gdst(i, j);

            tile_shape td0(2*i+j), td1(i+2*j), td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TAND_Impl(td2, td1, td0);
            // TCOPYOUT(g2, td2);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tand_nz(int *dst, int *src0, int *src1) {
    using gm_shape = global_tensor<int, RowMajor<gm_row, gm_col>>;
    using tile_shape = TileLeft<int, tile_row, tile_col>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc0(src0);
    iter gsrc1(src1);
    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(i, j);
            auto g1 = gsrc1(i, j);
            auto g2 = gdst(i, j);

            // tile_shape td0(2*i+j), td1(i+2*j), td2;
            //TAND_Impl(td2, td1, td0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tand_zn(int *dst, int *src0, int *src1) {
    using gm_shape = global_tensor<int, ColMajor<gm_row, gm_col>>;
    using tile_shape = TileRight<int, tile_row, tile_col>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc0(src0);
    iter gsrc1(src1);
    iter gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(i, j);
            auto g1 = gsrc1(i, j);
            auto g2 = gdst(i, j);

            // tile_shape td0(2*i+j), td1(i+2*j), td2;
            // TAND_Impl(td2, td1, td0);
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

    int src0[gm_size];
    int src1[gm_size];
    int dst[gm_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        tand_nd<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }else if(!strcmp(MODE, "NZ")){
        tand_nz<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }else if(!strcmp(MODE, "ZN")){
        tand_zn<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}