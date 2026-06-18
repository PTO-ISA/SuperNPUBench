#include <common/pto_tileop.hpp>
#include <string> 

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifndef ROW
#define ROW 33
#endif

#ifndef COL
#define COL 33
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
void tadd_mask_nd(float *dst, float *src0, float *src1) {
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;

    static constexpr int block_row = gm_row / tile_row;
    static constexpr int block_col = gm_col / tile_col;
    static constexpr int remainder_row = gm_row % tile_row;
    static constexpr int remainder_col = gm_col % tile_col;

    using trailing_rows_shape =
        Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor, tile_row, remainder_col>;
    using trailing_cols_shape =
        Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor, remainder_row, tile_col>;
    using trailing_corner_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor,
                                                remainder_row, remainder_col>;

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
            TADD(td2, td0, td1);
            // TCOPYOUT(g2, td2);
        }
        if constexpr (remainder_col) {
            auto g0 = gsrc0(i, block_col);
            auto g1 = gsrc1(i, block_col);
            auto g2 = gdst(i, block_col);

            trailing_rows_shape td0(2*i), td1(i), td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TADD(td2, td0, td1);
            // TCOPYOUT(g2, td2);
        }
    }
    if constexpr (remainder_row) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(block_row, j);
            auto g1 = gsrc1(block_row, j);
            auto g2 = gdst(block_row, j);

            trailing_cols_shape td0(j), td1(2*j), td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TADD(td2, td0, td1);
            // TCOPYOUT(g2, td2);
        }
        if constexpr (remainder_col) {
            auto g0 = gsrc0(block_row, block_col);
            auto g1 = gsrc1(block_row, block_col);
            auto g2 = gdst(block_row, block_col);

            trailing_corner_shape td0(0), td1(1), td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TADD(td2, td0, td1);
            // TCOPYOUT(g2, td2);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tadd_mask_nz(float *dst, float *src0, float *src1) {
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = TileLeft<float, tile_row, tile_col>;

    static constexpr int block_row = gm_row / tile_row;
    static constexpr int block_col = gm_col / tile_col;
    static constexpr int remainder_row = gm_row % tile_row;
    static constexpr int remainder_col = gm_col % tile_col;

}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tadd_mask_zn(float *dst, float *src0, float *src1) {
    using gm_shape = global_tensor<float, ColMajor<gm_row, gm_col>>;
    using tile_shape = TileRight<float, tile_row, tile_col>;

    static constexpr int block_row = gm_row / tile_row;
    static constexpr int block_col = gm_col / tile_col;
    static constexpr int remainder_row = gm_row % tile_row;
    static constexpr int remainder_col = gm_col % tile_col;
}

int main() {
    const uint16_t gm_row = ROW;
    const uint16_t gm_col = COL;
    const uint16_t tile_row = TROW;
    const uint16_t tile_col = TCOL;

    size_t gm_size = gm_row * gm_col;
    size_t tile_size = tile_row * tile_col;

    float src0[gm_size];
    float src1[gm_size];
    float dst[gm_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        tadd_mask_nd<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }else if(!strcmp(MODE, "NZ")){
        tadd_mask_nz<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }else if(!strcmp(MODE, "ZN")){
        tadd_mask_zn<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}