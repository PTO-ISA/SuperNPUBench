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
void tgather_nd(float *dst, float *src, uint16_t *indices) {
  using gm_shape_src = global_tensor<float, RowMajor<gm_row, gm_col>>;
  using gm_shape_indices = global_tensor<uint16_t, RowMajor<tile_row, tile_col>>;
  using gm_shape_dst = global_tensor<float, RowMajor<tile_row, tile_col>>;

  using tile_shape_src = Tile<Location::Vec, float, gm_row, gm_col, BLayout::RowMajor>;
  using tile_shape_indices = Tile<Location::Vec, uint16_t, tile_row, tile_col, BLayout::RowMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_src = global_iterator<gm_shape_src, tile_shape_src>;
    using iter_indices = global_iterator<gm_shape_indices, tile_shape_indices>;
    using iter_dst = global_iterator<gm_shape_dst, tile_shape_dst>;

    iter_src gsrc(src);
    iter_indices gindices(indices);
    iter_dst gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc(i, j);
            auto g1 = gindices(i, j);
            auto g2 = gdst(i, j);

            tile_shape_src td0(2*i+j);
            tile_shape_indices td1(1);
            tile_shape_dst td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TGATHER(td2, td0, td1);
            // TCOPYOUT(g2, td2);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tgather_dn(float *dst, float *src, uint16_t *indices) {
  using gm_shape_src = global_tensor<float, ColMajor<gm_row, gm_col>>;
  using gm_shape_indices = global_tensor<uint16_t, ColMajor<tile_row, tile_col>>;
  using gm_shape_dst = global_tensor<float, ColMajor<tile_row, tile_col>>;

  using tile_shape_src = Tile<Location::Vec, float, gm_row, gm_col, BLayout::ColMajor>;
  using tile_shape_indices = Tile<Location::Vec, uint16_t, tile_row, tile_col, BLayout::ColMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, tile_row, tile_col, BLayout::ColMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    using iter_src = global_iterator<gm_shape_src, tile_shape_src>;
    using iter_indices = global_iterator<gm_shape_indices, tile_shape_indices>;
    using iter_dst = global_iterator<gm_shape_dst, tile_shape_dst>;

    iter_src gsrc(src);
    iter_indices gindices(indices);
    iter_dst gdst(dst);

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            auto g0 = gsrc(i, j);
            auto g1 = gindices(i, j);
            auto g2 = gdst(i, j);

            tile_shape_src td0(2*i+j);
            tile_shape_indices td1(1);
            tile_shape_dst td2;
            // TCOPYIN(td0, g0);
            // TCOPYIN(td1, g1);
            TGATHER(td2, td0, td1);
            // TCOPYOUT(g2, td2);
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

    float dst[tile_size];
    float src[gm_size];
    uint16_t indices[tile_size];

    #ifdef LINX_PMC
    PMC_START();
    #endif

    if(!strcmp(MODE, "ND")){
        tgather_nd<gm_row, gm_col, tile_row, tile_col>(dst, src, indices);
    }else if(!strcmp(MODE, "DN")){
        tgather_dn<gm_row, gm_col, tile_row, tile_col>(dst, src, indices);
    }

    #ifdef LINX_PMC
    PMC_END();
    #endif

    return 0;
}