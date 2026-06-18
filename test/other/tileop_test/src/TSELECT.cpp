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

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tselect_nd(float *dst, float *src0, float *src1, uint16_t *cond) {
    using gm_shape_fp32 = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using gm_shape_uint16 = global_tensor<uint16_t, RowMajor<gm_row, gm_col>>;

    using tile_shape_fp32 = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;
    using tile_shape_uint16 = Tile<Location::Vec, uint16_t, tile_row, tile_col, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_fp32 = global_iterator<gm_shape_fp32, tile_shape_fp32>;
    using iter_uint16 = global_iterator<gm_shape_uint16, tile_shape_uint16>;

    iter_fp32 gsrc0(src0);
    iter_fp32 gsrc1(src1);
    iter_uint16 gcond(cond);
    iter_fp32 gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(i, j);
            auto g1 = gsrc1(i, j);
            auto g2 = gcond(i, j);
            auto g3 = gdst(i, j);

            tile_shape_fp32 td0(2*i+j), td1(i+2*j);
            tile_shape_uint16 td2(i%2);
            tile_shape_fp32 td3;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            // TCOPYIN(td2, g2);
            TSELECT(td3, td2, td0, td1);
            // TCOPYOUT(g3, td3);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tselect_dn(float *dst, float *src0, float *src1, uint16_t *cond) {
    using gm_shape_fp32 = global_tensor<float, ColMajor<gm_row, gm_col>>;
    using gm_shape_uint16 = global_tensor<uint16_t, ColMajor<gm_row, gm_col>>;

    using tile_shape_fp32 = Tile<Location::Vec, float, tile_row, tile_col, BLayout::ColMajor>;
    using tile_shape_uint16 = Tile<Location::Vec, uint16_t, tile_row, tile_col, BLayout::ColMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_fp32 = global_iterator<gm_shape_fp32, tile_shape_fp32>;
    using iter_uint16 = global_iterator<gm_shape_uint16, tile_shape_uint16>;

    iter_fp32 gsrc0(src0);
    iter_fp32 gsrc1(src1);
    iter_uint16 gcond(cond);
    iter_fp32 gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc0(i, j);
            auto g1 = gsrc1(i, j);
            auto g2 = gcond(i, j);
            auto g3 = gdst(i, j);

            tile_shape_fp32 td0(2*i+j), td1(i+2*j);
            tile_shape_uint16 td2(i%2);
            tile_shape_fp32 td3;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            // TCOPYIN(td2, g2);
            TSELECT(td3, td2, td0, td1);
            // TCOPYOUT(g3, td3);
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

    float dst[gm_size];
    float src0[gm_size];
    float src1[gm_size];
    uint16_t cond[tile_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        tselect_nd<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1, cond);
    }else if(!strcmp(MODE, "DN")){
        tselect_dn<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1, cond);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}