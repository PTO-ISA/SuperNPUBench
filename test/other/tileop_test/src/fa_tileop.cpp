#include <common/pto_tileop.hpp>

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

#ifdef DTYPE
    #if DTYPE == "float64"
        typedef double dtype;
    #elif DTYPE == "float32"
        typedef float dtype;
    #elif DTYPE == "int8"
        typedef int8_t dtype;
    #elif DTYPE == "int16"
        typedef int16_t dtype;
    #elif DTYPE == "int32"
        typedef int32_t dtype;
    #elif DTYPE == "int64"
        typedef int64_t dtype;
    #else
        typedef float dtype;
    #endif
#else
    typedef float dtype;
#endif  

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tsub_nz_left(dtype *dst, dtype *src0, dtype *src1) {
    using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape = TileLeft<dtype, tile_row, tile_col>;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc0(src0);
    iter gsrc1(src1);
    iter gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape tsrc0,tsrc1;
            TCOPYIN(tsrc0, gsrc0(i, j));
            TCOPYIN(tsrc1, gsrc1(i, j));
            TSUB(tsrc0, tsrc0, tsrc1);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, tsrc0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void trowsum_nz_left(dtype *dst, dtype *src) {
    using gm_shape_in = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape_in = TileLeft<dtype, tile_row, tile_col>;

    using tile_shape_r = Tile<Location::Vec, dtype, tile_row, 1, BLayout::RowMajor>;
    using gm_shape_out = global_tensor<dtype, RowMajor<gm_row, 1>>;

    using iterSrc = global_iterator<gm_shape_in, tile_shape_in>;
    using iterDst = global_iterator<gm_shape_out, tile_shape_r>;

    iterSrc gsrc(src);
    iterDst gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;
    for (int i = 0; i < block_row; ++i) {
        tile_shape_r trsum;
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in tsrc;
            TCOPYIN(tsrc, gsrc(i, j));
            TROWSUM(trsum, tsrc);
        }
        auto gO = gdst(i, 0);
        TCOPYOUT(gO, trsum);
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void texpandcol_nz_left(dtype *dst, dtype *src) {
    using gm_shape_in = global_tensor<dtype, RowMajor<gm_row, 1>>;
    using tile_shape_in = Tile<Location::Vec, dtype, tile_row, 1, BLayout::RowMajor>;

    using tile_shape_expand = TileLeft<dtype, tile_row, tile_col>;
    using gm_shape_out = global_tensor<dtype, RowMajor<gm_row, gm_col>>;

    using iterSrc = global_iterator<gm_shape_in, tile_shape_in>;
    using iterDst = global_iterator<gm_shape_out, tile_shape_expand>;

    iterSrc gsrc(src);
    iterDst gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in tsrc;
            tile_shape_expand texpand;
            TCOPYIN(tsrc, gsrc(i, 0));
            TEXPANDCOL(texpand, tsrc);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, texpand);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tmul_nz_out(dtype *dst, dtype *src0, dtype *src1) {
    using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape = TileAcc<dtype, tile_row, tile_col>;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc0(src0);
    iter gsrc1(src1);
    iter gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape tsrc0,tsrc1;
            TCOPYIN(tsrc0, gsrc0(i, j));
            TCOPYIN(tsrc1, gsrc1(i, j));
            TMUL(tsrc0, tsrc0, tsrc1);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, tsrc0);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void texpandcol_nz_out(dtype *dst, dtype *src) {
    using gm_shape_in = global_tensor<dtype, RowMajor<gm_row, 1>>;
    using tile_shape_in = Tile<Location::Vec, dtype, tile_row, 1, BLayout::RowMajor>;

    using tile_shape_expand = TileAcc<dtype, tile_row, tile_col>;
    using gm_shape_out = global_tensor<dtype, RowMajor<gm_row, gm_col>>;

    using iterSrc = global_iterator<gm_shape_in, tile_shape_in>;
    using iterDst = global_iterator<gm_shape_out, tile_shape_expand>;

    iterSrc gsrc(src);
    iterDst gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in tsrc;
            tile_shape_expand texpand;
            TCOPYIN(tsrc, gsrc(i, 0));
            TEXPANDCOL(texpand, tsrc);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, texpand);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void trowmax_nz_left(dtype *dst, dtype *src) {
    using gm_shape_in = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape_in = TileLeft<dtype, tile_row, tile_col>;

    using tile_shape_r = Tile<Location::Vec, dtype, tile_row, 1, BLayout::RowMajor>;
    using gm_shape_out = global_tensor<dtype, RowMajor<gm_row, 1>>;

    using iterSrc = global_iterator<gm_shape_in, tile_shape_in>;
    using iterDst = global_iterator<gm_shape_out, tile_shape_r>;

    iterSrc gsrc(src);
    iterDst gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        tile_shape_r trmax;
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in tsrc;
            TCOPYIN(tsrc, gsrc(i, j));
            TROWMAX(trmax, tsrc);
        }
        auto gO = gdst(i, 0);
        TCOPYOUT(gO, trmax);
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tmuls_nz_left(dtype *dst, dtype *src, dtype s) {
    using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape = TileLeft<dtype, tile_row, tile_col>;

    using iter = global_iterator<gm_shape, tile_shape>;

    iter gsrc(src);
    iter gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape tsrc;
            TCOPYIN(tsrc, gsrc(i, j));
            TMULS(tsrc, tsrc, s);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, tsrc);
        }
    }
}

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row, uint16_t tile_col>
void tcvt_out_left(dtype *dst, dtype *src) {
    using gm_shape = global_tensor<dtype, RowMajor<gm_row, gm_col>>;
    using tile_shape_in = TileLeft<dtype, tile_row, tile_col>;
    using tile_shape_out = TileAcc<dtype, tile_row, tile_col>;

    using iter = global_iterator<gm_shape, tile_shape_in>;

    iter gsrc(src);
    iter gdst(dst);

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            tile_shape_in tsrc;
            tile_shape_out tout;
            TCOPYIN(tsrc, gsrc(i, j));
            TCVT(tout, tsrc);
            auto gO = gdst(i, j);
            TCOPYOUT(gO, tout);
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

    dtype src0[gm_size];
    dtype src1[gm_size];
    dtype dst[gm_size];

    if(!strcmp(MODE, "tcvt_out_left")){
        tcvt_out_left<gm_row, gm_col, tile_row, tile_col>(dst, src0);
    }else if(!strcmp(MODE, "tmuls_nz_left")){
        tmuls_nz_left<gm_row, gm_col, tile_row, tile_col>(dst, src0, 56.7f);
    }else if(!strcmp(MODE, "trowmax_nz_left")){
        trowmax_nz_left<gm_row, gm_col, tile_row, tile_col>(dst, src0);
    }else if(!strcmp(MODE, "texpandcol_nz_out")){
        texpandcol_nz_out<gm_row, gm_col, tile_row, tile_col>(dst, src0);
    }else if(!strcmp(MODE, "tmul_nz_out")){
        tmul_nz_out<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }else if(!strcmp(MODE, "texpandcol_nz_left")){
        texpandcol_nz_left<gm_row, gm_col, tile_row, tile_col>(dst, src0);
    }else if(!strcmp(MODE, "trowsum_nz_left")){
        trowsum_nz_left<gm_row, gm_col, tile_row, tile_col>(dst, src0);
    }else if(!strcmp(MODE, "tsub_nz_left")){
        tsub_nz_left<gm_row, gm_col, tile_row, tile_col>(dst, src0, src1);
    }

    return 0;
}